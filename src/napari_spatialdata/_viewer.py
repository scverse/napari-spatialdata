from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
from anndata import AnnData
from loguru import logger
from napari import Viewer
from napari.layers import Labels, Points, Shapes
from napari.utils.notifications import show_info

from napari_spatialdata.utils._utils import (
    _adjust_channels_order,
    _get_transform,
    _swap_coordinates,
    get_duplicate_element_names,
)

if TYPE_CHECKING:
    from napari.layers import Layer
    from napari.utils.events import Event, EventedList
    from spatialdata import SpatialData


class SpatialDataViewer:
    def __init__(self, viewer: Viewer, sdata: EventedList) -> None:
        self.viewer = viewer
        self.sdata = sdata
        self._layer_action_caches: dict[str, list[dict[str, Any]]] = {}
        self.viewer.bind_key("Shift-L", self._inherit_metadata)
        self.viewer.layers.events.inserted.connect(self._on_layer_insert)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)

        # Used to check old layer name. This because event emitted does not contain this information.
        self.layer_names: set[str | None] = set()

    def _on_layer_insert(self, event: Event) -> None:
        layer = event.value
        self.layer_names.add(layer.name)
        self._layer_action_caches[layer.name] = []
        layer.events.data.connect(self._update_cache)
        layer.events.name.connect(self._validate_name)

    def _on_layer_removed(self, event: Event) -> None:
        layer = event.value
        del self._layer_action_caches[layer.name]
        self.layer_names.remove(layer.name)

    def _validate_name(self, event: Event) -> None:
        _, element_names = get_duplicate_element_names(self.sdata)
        current_layer_names = [layer.name for layer in self.viewer.layers]
        old_layer_name = self.layer_names.difference(current_layer_names).pop()

        layer = event.source
        sdata = layer.metadata.get("sdata")

        pattern = r" \[\d+\]$"
        duplicate_pattern_found = re.search(pattern, layer.name)
        name_to_validate = re.sub(pattern, "", layer.name) if duplicate_pattern_found else layer.name

        # Ensures that the callback does not get called a second time when changing layer.name here.
        with layer.events.name.blocker(self._validate_name):
            if sdata:
                sdata_names = [element_name for _, element_name, _ in sdata._gen_elements()]
                if name_to_validate in sdata_names or duplicate_pattern_found:
                    layer.name = old_layer_name
                    show_info("New layer name causes name conflicts. Reverting to old layer name")
                elif name_to_validate in element_names:
                    sdata_index = self.sdata.index(sdata)
                    layer.name = name_to_validate + f"_{sdata_index}"
            elif duplicate_pattern_found or name_to_validate in element_names:
                layer.name = old_layer_name
                show_info("Layer name potentially causes naming conflicts with SpatialData elements. Reverting.")

        self.layer_names.remove(old_layer_name)
        self.layer_names.add(layer.name)

    def _update_cache(self, event: Event) -> None:
        event_info = {
            "data_indices": event.data_indices,
            "vertex_indices": event.vertex_indices,
            "action": event.action,
        }
        layer_name = event.source.name
        self._layer_action_caches[layer_name].append(event_info)

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

    def add_sdata_image(self, sdata: SpatialData, key: str, selected_cs: str, multi: bool) -> None:
        original_name = key
        if multi:
            original_name = original_name[: original_name.rfind("_")]

        affine = _get_transform(sdata.images[original_name], selected_cs)
        rgb_image, rgb = _adjust_channels_order(element=sdata.images[original_name])

        # TODO: type check
        self.viewer.add_image(
            rgb_image,
            rgb=rgb,
            name=key,
            affine=affine,
            metadata={
                "sdata": sdata,
                "name": original_name,
                "_active_in_cs": {selected_cs},
                "_current_cs": selected_cs,
            },
        )

    def add_sdata_circles(self, sdata: SpatialData, key: str, selected_cs: str, multi: bool) -> None:
        original_name = key
        if multi:
            original_name = original_name[: original_name.rfind("_")]

        df = sdata.shapes[original_name]
        affine = _get_transform(sdata.shapes[original_name], selected_cs)

        xy = np.array([df.geometry.x, df.geometry.y]).T
        xy = np.fliplr(xy)
        radii = df.radius.to_numpy()

        self.viewer.add_points(
            xy,
            name=key,
            affine=affine,
            size=2 * radii,
            edge_width=0.0,
            metadata={
                "sdata": sdata,
                "adata": sdata.table[
                    sdata.table.obs[sdata.table.uns["spatialdata_attrs"]["region_key"]] == original_name
                ],
                "region_key": sdata.table.uns["spatialdata_attrs"]["region_key"],
                "name": original_name,
                "_active_in_cs": {selected_cs},
                "_current_cs": selected_cs,
            },
        )

    def add_sdata_shapes(self, sdata: SpatialData, key: str, selected_cs: str, multi: bool) -> None:
        original_name = key
        if multi:
            original_name = original_name[: original_name.rfind("_")]

        polygons = []
        df = sdata.shapes[original_name]
        affine = _get_transform(sdata.shapes[original_name], selected_cs)

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
                "name": original_name,
                "_active_in_cs": {selected_cs},
                "_current_cs": selected_cs,
            },
        )

    def add_sdata_labels(self, sdata: SpatialData, key: str, selected_cs: str, multi: bool) -> None:
        original_name = key
        if multi:
            original_name = original_name[: original_name.rfind("_")]

        affine = _get_transform(sdata.labels[original_name], selected_cs)
        rgb_labels, _ = _adjust_channels_order(element=sdata.labels[original_name])

        self.viewer.add_labels(
            rgb_labels,
            name=key,
            affine=affine,
            metadata={
                "sdata": sdata,
                "adata": sdata.table[sdata.table.obs[sdata.table.uns["spatialdata_attrs"]["region_key"]] == key],
                "region_key": sdata.table.uns["spatialdata_attrs"]["instance_key"],
                "name": original_name,
                "_active_in_cs": {selected_cs},
                "_current_cs": selected_cs,
            },
        )

    def add_sdata_points(self, sdata: SpatialData, key: str, selected_cs: str, multi: bool) -> None:
        original_name = key
        if multi:
            original_name = original_name[: original_name.rfind("_")]

        points = sdata.points[original_name].compute()
        affine = _get_transform(sdata.points[original_name], selected_cs)
        if len(points) < 100000:
            subsample = np.arange(len(points))
        else:
            logger.info("Subsampling points because the number of points exceeds the currently supported 100 000.")
            gen = np.random.default_rng()
            subsample = np.sort(gen.choice(len(points), size=100000, replace=False))

        xy = points[["y", "x"]].values[subsample]
        np.fliplr(xy)
        self.viewer.add_points(
            xy,
            name=key,
            size=20,
            affine=affine,
            edge_width=0.0,
            metadata={
                "sdata": sdata,
                "adata": AnnData(obs=points.iloc[subsample, :], obsm={"spatial": xy}),
                "name": original_name,
                "_active_in_cs": {selected_cs},
                "_current_cs": selected_cs,
            },
        )

    def _affine_transform_layers(self, coordinate_system: str) -> None:
        for layer in self.viewer.layers:
            metadata = layer.metadata
            if metadata.get("sdata"):
                sdata = metadata["sdata"]
                element_name = metadata["name"]
                element_data = sdata[element_name]
                affine = _get_transform(element_data, coordinate_system)
                if affine is not None:
                    layer.affine = affine
