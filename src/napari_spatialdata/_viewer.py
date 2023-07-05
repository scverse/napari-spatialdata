from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from anndata import AnnData
from loguru import logger
from napari import Viewer
from napari.layers import Labels, Points, Shapes
from napari.utils.notifications import show_info

from napari_spatialdata.utils._utils import _adjust_channels_order, _get_transform, _swap_coordinates

if TYPE_CHECKING:
    from napari.layers import Layer
    from spatialdata import SpatialData


class SpatialDataViewer:
    def __init__(self, viewer: Viewer) -> None:
        self.viewer = viewer
        self.viewer.bind_key("Shift-L", self._inherit_metadata)

    def _inherit_metadata(self, viewer: Viewer) -> None:
        layers = list(viewer.layers.selection)
        self.inherit_metadata(layers)

    def inherit_metadata(self, layers: list[Layer]) -> None:
        """
        Inherit metadata from active layer.

        A new layer that is added will inherit from the layer that is active when its added, ensuring proper association
        with a spatialdata object and coordinate space.

        Parameters
        ----------
        layers: list[Layer]
            A list of napari layers of which only 1 should have a spatialdata object from which the other layers inherit
            metadata.
        """
        # Layer.metadata.get would yield a default value which is not what we want.
        sdatas = [layer.metadata["sdata"] for layer in layers if "sdata" in layer.metadata]

        # If more than 1 sdata object, ensure all are the same.
        if len(sdatas) > 1 and not all(sdatas[0] is sdata for sdata in sdatas[1:]):
            raise ValueError("Multiple different spatialdata object found in selected layers. One is required.")

        if len(sdatas) < 1:
            raise ValueError("No Spatialdata objects associated with selected layers.")

        ref_layer = next(layer for layer in layers if "sdata" in layer.metadata)

        for layer in (
            layer
            for layer in layers
            if layer != ref_layer and isinstance(layer, (Labels, Points, Shapes)) and "sdata" not in layer.metadata
        ):
            layer.metadata["sdata"] = ref_layer.metadata["sdata"]
            layer.metadata["_current_cs"] = ref_layer.metadata["_current_cs"]
            layer.metadata["_active_in_cs"] = {ref_layer.metadata["_current_cs"]}
            layer.metadata["name"] = None
            layer.metadata["adata"] = None
            if isinstance(layer, (Shapes, Labels)):
                layer.metadata["region_key"] = None

        show_info(f"Layer(s) without associated SpatialData object inherited SpatialData metadata of {ref_layer}")

    def add_sdata_image(self, sdata: SpatialData, selected_cs: str, key: str) -> None:
        affine = _get_transform(sdata.images[key], selected_cs)
        rgb_image, rgb = _adjust_channels_order(element=sdata.images[key])

        # TODO: type check
        self.viewer.add_image(
            rgb_image,
            rgb=rgb,
            name=key,
            affine=affine,
            metadata={"sdata": sdata, "_active_in_cs": {selected_cs}, "_current_cs": selected_cs},
        )

    def add_sdata_circles(self, sdata: SpatialData, selected_cs: str, key: str) -> None:
        df = sdata.shapes[key]
        affine = _get_transform(sdata.shapes[key], selected_cs)

        xy = np.array([df.geometry.x, df.geometry.y]).T
        xy = np.fliplr(xy)
        radii = np.array([df.radius[i] for i in range(0, len(df))])

        self.viewer.add_points(
            xy,
            name=key,
            affine=affine,
            size=2 * radii,
            edge_width=0.0,
            metadata={
                "sdata": sdata,
                "adata": sdata.table[sdata.table.obs[sdata.table.uns["spatialdata_attrs"]["region_key"]] == key],
                "region_key": sdata.table.uns["spatialdata_attrs"]["region_key"],
                "name": key,
                "_active_in_cs": {selected_cs},
                "_current_cs": selected_cs,
            },
        )

    def add_sdata_shapes(self, sdata: SpatialData, selected_cs: str, key: str) -> None:
        polygons = []
        df = sdata.shapes[key]
        affine = _get_transform(sdata.shapes[key], selected_cs)

        # when mulitpolygons are present, we select the largest ones
        if "MultiPolygon" in np.unique(df.geometry.type):
            logger.info("Multipolygons are present in the data. Only the largest polygon per cell is retained.")
            df = df.explode(index_parts=False)
            df["area"] = df.area
            df = df.sort_values(by="area", ascending=False)  # sort by area
            df = df[~df.index.duplicated(keep="first")]  # only keep the largest area
            df = df.sort_index()  # reset the index to the first order
        if len(df) < 100:
            for i in range(0, len(df)):
                polygons.append(list(df.geometry.iloc[i].exterior.coords))
        else:
            for i in range(
                0, len(df)
            ):  # This can be removed once napari is sped up in the plotting. It changes the shapes only very slightly
                polygons.append(list(df.geometry.iloc[i].exterior.simplify(tolerance=2).coords))
        # this will only work for polygons and not for multipolygons
        polygons = _swap_coordinates(polygons)

        self.viewer.add_shapes(
            polygons,
            name=key,
            affine=affine,
            shape_type="polygon",
            metadata={
                "sdata": sdata,
                "adata": sdata.table[sdata.table.obs[sdata.table.uns["spatialdata_attrs"]["region_key"]] == key],
                "region_key": sdata.table.uns["spatialdata_attrs"]["region_key"],
                "name": key,
                "_active_in_cs": {selected_cs},
                "_current_cs": selected_cs,
            },
        )

    def add_sdata_labels(self, sdata: SpatialData, selected_cs: str, key: str) -> None:
        affine = _get_transform(sdata.labels[key], selected_cs)

        rgb_labels, _ = _adjust_channels_order(element=sdata.labels[key])

        self.viewer.add_labels(
            rgb_labels,
            name=key,
            affine=affine,
            metadata={
                "sdata": sdata,
                "adata": sdata.table[sdata.table.obs[sdata.table.uns["spatialdata_attrs"]["region_key"]] == key],
                "region_key": sdata.table.uns["spatialdata_attrs"]["instance_key"],
                "name": key,
                "_active_in_cs": {selected_cs},
                "_current_cs": selected_cs,
            },
        )

    def add_sdata_points(self, sdata: SpatialData, selected_cs: str, key: str) -> None:
        points = sdata.points[key].compute()
        affine = _get_transform(sdata.points[key], selected_cs)
        if len(points) < 100000:
            subsample = np.arange(len(points))
        else:
            logger.info("Subsampling points because the number of points exceeds the currently supported 100 000.")
            gen = np.random.default_rng()
            subsample = gen.choice(len(points), size=100000, replace=False)

        self.viewer.add_points(
            points[["y", "x"]].values[subsample],
            name=key,
            size=20,
            affine=affine,
            edge_width=0.0,
            metadata={
                "sdata": sdata,
                "adata": AnnData(obs=points.loc[subsample, :], obsm={"spatial": points[["x", "y"]].values[subsample]}),
                "name": key,
                "_active_in_cs": {selected_cs},
                "_current_cs": selected_cs,
            },
        )
