from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from anndata import AnnData
from loguru import logger
from multiscale_spatial_image import MultiscaleSpatialImage
from napari import Viewer
from napari.layers import Labels, Points, Shapes
from napari.utils.notifications import show_info

from napari_spatialdata.utils._utils import _get_transform, _swap_coordinates

if TYPE_CHECKING:
    from napari.utils.events.event import Event
    from spatialdata import SpatialData


class SpatialDataViewer(Viewer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layers.events.inserted.connect(self._inherit_metadata)

    def _inherit_metadata(self, event: Event) -> None:
        """
        Inherit metadata from active layer.

        A new layer that is added will inherit from the layer that is active when its added, ensuring proper association
        with a spatialdata object and coordinate space.

        Paramters
        ---------
        event: Event
            A layer inserted event
        """
        layer = event.value
        active_layer = self.layers.selection.active

        if active_layer and type(layer) in {Labels, Points, Shapes} and "sdata" not in layer.metadata:
            active_layer_metadata = active_layer.metadata
            layer.metadata["sdata"] = active_layer_metadata["sdata"]
            layer.metadata["_current_cs"] = active_layer_metadata["_current_cs"]
            layer.metadata["_active_in_cs"] = {active_layer_metadata["_current_cs"]}
            show_info(f"The spatialdata object is set to the spatialdata object of {active_layer}")

    def add_sdata_image(self, sdata: SpatialData, selected_cs: str, key: str) -> None:
        img = sdata.images[key]
        affine = _get_transform(sdata.images[key], selected_cs)

        if isinstance(img, MultiscaleSpatialImage):
            img = img["scale0"][key]
        # TODO: type check
        self.add_image(
            img,
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

        self.add_points(
            xy,
            name=key,
            affine=affine,
            size=2 * radii,
            edge_width=0.0,
            metadata={
                "sdata": sdata,
                "adata": sdata.table[sdata.table.obs[sdata.table.uns["spatialdata_attrs"]["region_key"]] == key],
                "shapes_key": sdata.table.uns["spatialdata_attrs"]["region_key"],
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

        self.add_shapes(
            polygons,
            name=key,
            affine=affine,
            shape_type="polygon",
            metadata={
                "sdata": sdata,
                "adata": sdata.table[sdata.table.obs[sdata.table.uns["spatialdata_attrs"]["region_key"]] == key],
                "shapes_key": sdata.table.uns["spatialdata_attrs"]["region_key"],
                "shapes_type": "polygons",
                "name": key,
                "_active_in_cs": {selected_cs},
                "_current_cs": selected_cs,
            },
        )

    def add_sdata_labels(self, sdata: SpatialData, selected_cs: str, key: str) -> None:
        affine = _get_transform(sdata.labels[key], selected_cs)

        self.add_labels(
            sdata.labels[key],
            name=key,
            affine=affine,
            metadata={
                "sdata": sdata,
                "adata": sdata.table[sdata.table.obs[sdata.table.uns["spatialdata_attrs"]["region_key"]] == key],
                "labels_key": sdata.table.uns["spatialdata_attrs"]["instance_key"],
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

        self.add_points(
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
