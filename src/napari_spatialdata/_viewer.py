from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
from anndata import AnnData
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from loguru import logger
from napari import Viewer
from napari.layers import Image, Labels, Points, Shapes
from napari.utils.notifications import show_info
from qtpy.QtCore import QObject, Signal
from shapely import Polygon
from spatialdata._core.query.relational_query import (
    _get_element_annotators,
    _get_unique_label_values_as_index,
    _left_join_spatialelement_table,
)
from spatialdata.models import PointsModel, ShapesModel, force_2d
from spatialdata.transformations import Affine, Identity
from spatialdata.transformations._utils import scale_radii

from napari_spatialdata._constants import config
from napari_spatialdata.utils._utils import (
    _adjust_channels_order,
    _get_ellipses_from_circles,
    _get_init_metadata_adata,
    _get_transform,
    _transform_coordinates,
    get_duplicate_element_names,
)
from napari_spatialdata.utils._viewer_utils import _get_polygons_properties

if TYPE_CHECKING:
    import numpy.typing as npt
    from napari.layers import Layer
    from napari.utils.events import Event, EventedList
    from spatialdata import SpatialData


class SpatialDataViewer(QObject):
    layer_saved = Signal(object)

    def __init__(self, viewer: Viewer, sdata: EventedList) -> None:
        super().__init__()
        self.viewer = viewer
        self.sdata = sdata
        self._layer_event_caches: dict[str, list[dict[str, Any]]] = {}
        self.viewer.bind_key("Shift-L", self._inherit_metadata, overwrite=True)
        self.viewer.bind_key("Shift-E", self._save_to_sdata, overwrite=True)
        self.viewer.layers.events.inserted.connect(self._on_layer_insert)
        self._active_layer_table_names = None
        self.viewer.layers.events.removed.connect(self._on_layer_removed)

        # Used to check old layer name. This because event emitted does not contain this information.
        self.layer_names: set[str | None] = set()

    def _on_layer_insert(self, event: Event) -> None:
        layer = event.value
        if layer.metadata.get("sdata"):
            self.layer_names.add(layer.name)
            self._layer_event_caches[layer.name] = []
            layer.events.data.connect(self._update_cache_indices)
            layer.events.name.connect(self._validate_name)

    def _on_layer_removed(self, event: Event) -> None:
        layer = event.value
        if layer.metadata.get("name"):
            del self._layer_event_caches[layer.name]
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

    def _update_cache_indices(self, event: Event) -> None:
        del event.value
        # This needs to be changed when we switch to napari 0.4.19
        if event.action == "remove" or (type(event.source) != Points and event.action == "change"):
            # We overwrite the indices so they correspond to indices in the dataframe
            napari_indices = sorted(event.data_indices, reverse=True)
            event.indices = tuple(event.source.metadata["indices"][i] for i in napari_indices)
            if event.action == "remove":
                for i in napari_indices:
                    del event.source.metadata["indices"][i]
        elif type(event.source) == Points and event.action == "change":
            logger.warning(
                "Moving events of Points in napari can't be cached due to a bug in napari 0.4.18. This will"
                "be available in napari 0.4.19"
            )
            return
        if event.action == "add":
            # we need to add based on the indices of the dataframe, which can be subsampled in case of points
            n_indices = event.source.metadata["_n_indices"]
            event.indices = tuple(n_indices + i for i in range(len(event.data_indices)))
            event.source.metadata["_n_indices"] = event.indices[-1] + 1
            event.source.metadata["indices"].extend(event.indices)

        layer_name = event.source.name
        self._layer_event_caches[layer_name].append(event)

    def _save_to_sdata(self, viewer: Viewer) -> None:
        layer_selection = list(viewer.layers.selection)
        self.save_to_sdata(layer_selection)

    def save_to_sdata(self, layers: list[Layer]) -> None:
        """
        Add the current selected napari layer(s) to the SpatialData object.

        If the layer is newly added and not yet linked with a spatialdata object it will be automatically
        linked if only 1 spatialdata object is being visualized in the viewer.

        Notes
        -----
        Usage:

            - you can invoke this function by pressing Shift+E;
            - the selected layer (needs to be exactly one) will be saved;
            - if more than one SpatialData object is being shown with napari, before saving the layer you need to link
              it to a layer with a SpatialData object. This can be done by selecting both layers and pressing Shift+L.
            - Currently images and labels are not supported.
            - Currently updating existing elements is not supported.
        """
        # TODO: change the logic to match the new docstring

        selected_layers = self.viewer.layers.selection
        if len(selected_layers) != 1:
            raise ValueError("Only one layer can be saved at a time.")
        selected = list(selected_layers)[0]
        if "sdata" not in selected.metadata:
            sdatas = [(layer, layer.metadata["sdata"]) for layer in self.viewer.layers if "sdata" in layer.metadata]
            if len(sdatas) < 1:
                raise ValueError(
                    "No SpatialData layers found in the viewer. Layer cannot be linked to SpatialData object."
                )
            if len(sdatas) > 1 and not all(sdatas[0][1] is sdata[1] for sdata in sdatas[1:]):
                raise ValueError(
                    "Multiple different spatialdata object found in the viewer. Please link the layer to "
                    "one of them by selecting both the layer to save and the layer containing the SpatialData object "
                    "and then pressing Shift+L. Then select the layer to save and press Shift+E again."
                )
            # link the layer to the only sdata object
            self._inherit_metadata(self.viewer)
        assert selected.metadata["sdata"]

        # now we can save the layer since it is linked to a SpatialData object
        if not selected.metadata["name"]:
            sdata = selected.metadata["sdata"]
            coordinate_system = selected.metadata["_current_cs"]
            transformation = {coordinate_system: Identity()}
            swap_data: None | npt.ArrayLike
            if type(selected) == Points:
                if len(selected.data) == 0:
                    raise ValueError("Cannot export a points element with no points")
                transformed_data = np.array([selected.data_to_world(xy) for xy in selected.data])
                swap_data = np.fliplr(transformed_data)
                # ignore z axis if present
                if swap_data.shape[1] == 3:
                    swap_data = swap_data[:, :2]
                parsed = PointsModel.parse(swap_data, transformations=transformation)
                sdata.points[selected.name] = parsed
            if type(selected) == Shapes:
                if len(selected.data) == 0:
                    raise ValueError("Cannot export a shapes element with no shapes")
                polygons: list[Polygon] = [
                    Polygon(i) for i in _transform_coordinates(selected.data, f=lambda x: x[::-1])
                ]
                gdf = GeoDataFrame({"geometry": polygons})
                force_2d(gdf)
                parsed = ShapesModel.parse(gdf, transformations=transformation)
                sdata.shapes[selected.name] = parsed
                swap_data = None
            if type(selected) == Image or type(selected) == Labels:
                raise NotImplementedError

            self.layer_names.add(selected.name)
            self._layer_event_caches[selected.name] = []
            self._update_metadata(selected, parsed, swap_data)
            selected.events.data.connect(self._update_cache_indices)
            selected.events.name.connect(self._validate_name)
            self.layer_saved.emit(coordinate_system)
            show_info("Layer added to the SpatialData object")
        else:
            raise NotImplementedError("updating existing elements in-place will soon be supported")

    def _update_metadata(self, layer: Layer, model: DaskDataFrame, data: None | npt.ArrayLike = None) -> None:
        layer.metadata["name"] = layer.name
        layer.metadata["_n_indices"] = len(layer.data)
        layer.metadata["indices"] = list(i for i in range(len(layer.data)))  # noqa: C400
        if type(layer) == Points:
            layer.metadata["adata"] = AnnData(obs=model)

    def _get_layer_for_unique_sdata(self, viewer: Viewer) -> Layer:
        # If there is only one sdata object across all the layers, any layer containing the sdata object will be the
        # ref_layer. Otherwise, if multiple sdata object are available, the search will be restricted to the selected
        # layers. In all the other cases, i.e. multipe sdata objects in the selected layers, or zero sdata objects,
        # an exception will be raised.
        # check all layers
        sdatas = [(layer, layer.metadata["sdata"]) for layer in viewer.layers if "sdata" in layer.metadata]
        if len(sdatas) < 1:
            raise ValueError("No SpatialData layers found in the viewer. Layer cannot be linked to SpatialData object.")
        # If more than 1 sdata object, check whether all are the same. If not check layer selection
        if len(sdatas) > 1 and not all(sdatas[0][1] is sdata[1] for sdata in sdatas[1:]):
            # check only the selected layers
            layers = list(viewer.layers.selection)
            sdatas = [(layer, layer.metadata["sdata"]) for layer in layers if "sdata" in layer.metadata]
            if len(sdatas) > 1 and not all(sdatas[0][1] is sdata[1] for sdata in sdatas[1:]):
                raise ValueError("Multiple different spatialdata object found in selected layers. One is required.")
            if sdatas:
                ref_layer = sdatas[0][0]
            else:
                raise ValueError("Multiple SpatialData objects, but no layer with sdata in layer selection.")
        else:
            ref_layer = sdatas[0][0]
        return ref_layer

    def _inherit_metadata(self, viewer: Viewer, show_tooltip: bool = False) -> None:
        # This function calls inherit_metadata by setting a default value for ref_layer.
        layers = list(viewer.layers.selection)
        ref_layer = self._get_layer_for_unique_sdata(viewer)
        self.inherit_metadata(layers, ref_layer)

    def inherit_metadata(self, layers: list[Layer], ref_layer: Layer) -> None:
        """
        Inherit metadata from active layer.

        A new layer that is added will inherit from the layer that is active when its added, ensuring proper association
        with a spatialdata object and coordinate system.

        Parameters
        ----------
        layers: list[Layer]
            A list of napari layers. Layers already containing a `SpatialData` object in the metadata will be ignored;
            layers not containing it will inherit the metadata from the layer specified by the `ref_layer` argument.
        ref_layer: Layer
            The layer containing the `SpatialData` object in the metadata to which the layers will be linked
        """
        if not ref_layer.metadata.get("sdata"):
            raise ValueError(f"{ref_layer} does not contain a SpatialData object in the metadata. Can't link layers.")

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
                layer.metadata["instance_key"] = None
            if isinstance(layer, (Shapes, Points)):
                layer.metadata["_n_indices"] = None
                layer.metadata["indices"] = None

        show_info(f"Layer(s) inherited info from {ref_layer}")

    def _get_table_data(
        self, sdata: SpatialData, element_name: str
    ) -> tuple[AnnData | None, str | None, list[str | None]]:
        table_names = list(_get_element_annotators(sdata, element_name))
        table_name = table_names[0] if len(table_names) > 0 else None
        adata = _get_init_metadata_adata(sdata, table_name, element_name)
        return adata, table_name, table_names

    def add_sdata_image(self, sdata: SpatialData, key: str, selected_cs: str, multi: bool) -> None:
        """
        Add an image in a spatial data object to the viewer.

        Parameters
        ----------
        sdata
            The spatial data object containing the image.
        key
            The name of the image in the spatialdata object.
        selected_cs
            The coordinate system in which the image layer is to be loaded.
        multi
            Whether there are multiple spatialdata objects present in the viewer.
        """
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
        """
        Add a shapes layer to the viewer to visualize Point geometries.

        Parameters
        ----------
        sdata
            The spatial data object containing the Point geometries.
        key
            The name of the Shapes element in the spatialdata object.
        selected_cs
            The coordinate system in which the shapes layer is to be loaded.
        multi
            Whether there are multiple spatialdata objects present in the viewer.
        """
        original_name = key
        if multi:
            original_name = original_name[: original_name.rfind("_")]

        df = sdata.shapes[original_name]
        affine = _get_transform(sdata.shapes[original_name], selected_cs)

        xy = np.array([df.geometry.x, df.geometry.y]).T
        yx = np.fliplr(xy)
        radii = df.radius.to_numpy()

        adata, table_name, table_names = self._get_table_data(sdata, original_name)
        metadata = {
            "sdata": sdata,
            "adata": adata,
            "region_key": sdata[table_name].uns["spatialdata_attrs"]["region_key"] if table_name else None,
            "instance_key": sdata[table_name].uns["spatialdata_attrs"]["instance_key"] if table_name else None,
            "table_names": table_names if table_name else None,
            "name": original_name,
            "_active_in_cs": {selected_cs},
            "_current_cs": selected_cs,
            "_n_indices": len(df),
            "indices": df.index.to_list(),
            "_columns_df": (
                df_sub_columns if (df_sub_columns := df.drop(columns=["geometry", "radius"])).shape[1] != 0 else None
            ),
        }

        CIRCLES_AS_POINTS = True
        if CIRCLES_AS_POINTS:
            layer = self.viewer.add_points(
                yx,
                name=key,
                affine=affine,
                size=1,  # the sise doesn't matter here since it will be adjusted in _adjust_radii_of_points_layer
                edge_width=0.0,
                metadata=metadata,
            )
            assert affine is not None
            self._adjust_radii_of_points_layer(layer=layer, affine=affine)
        else:
            # useful code to have readily available to debug the correct radius of circles when represented as points
            ellipses = _get_ellipses_from_circles(yx=yx, radii=radii)
            self.viewer.add_shapes(
                ellipses,
                shape_type="ellipse",
                name=key,
                edge_color="white",
                face_color="white",
                edge_width=0.0,
                affine=affine,
                metadata=metadata,
            )

    def add_sdata_shapes(self, sdata: SpatialData, key: str, selected_cs: str, multi: bool) -> None:
        """
        Add shapes element in a spatial data object to the viewer.

        Parameters
        ----------
        sdata
            The spatial data object containing the shapes element.
        key
            The name of the shapes element in the spatialdata object.
        selected_cs
            The coordinate system in which the shapes element layer is to be loaded.
        multi
            Whether there are multiple spatialdata objects present in the viewer.
        """
        original_name = key
        if multi:
            original_name = original_name[: original_name.rfind("_")]

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

        simplify = len(df) > config.POLYGON_THRESHOLD
        polygons, indices = _get_polygons_properties(df, simplify)

        # this will only work for polygons and not for multipolygons
        polygons = _transform_coordinates(polygons, f=lambda x: x[::-1])

        adata, table_name, table_names = self._get_table_data(sdata, original_name)

        self.viewer.add_shapes(
            polygons,
            name=key,
            affine=affine,
            shape_type="polygon",
            metadata={
                "sdata": sdata,
                "adata": adata,
                "region_key": sdata[table_name].uns["spatialdata_attrs"]["region_key"] if table_name else None,
                "instance_key": sdata[table_name].uns["spatialdata_attrs"]["instance_key"] if table_name else None,
                "table_names": table_names if table_name else None,
                "name": original_name,
                "_active_in_cs": {selected_cs},
                "_current_cs": selected_cs,
                "_n_indices": len(df),
                "indices": indices,
                "_columns_df": (
                    df_sub_columns if (df_sub_columns := df.drop(columns="geometry")).shape[1] != 0 else None
                ),
            },
        )

    def add_sdata_labels(self, sdata: SpatialData, key: str, selected_cs: str, multi: bool) -> None:
        """
        Add a label element in a spatial data object to the viewer.

        Parameters
        ----------
        sdata
            The spatial data object containing the label element.
        key
            The name of the label element in the spatialdata object.
        selected_cs
            The coordinate system in which the labels layer is to be loaded.
        multi
            Whether there are multiple spatialdata objects present in the viewer.
        """
        original_name = key
        if multi:
            original_name = original_name[: original_name.rfind("_")]

        indices = _get_unique_label_values_as_index(sdata.labels[original_name])
        affine = _get_transform(sdata.labels[original_name], selected_cs)
        rgb_labels, _ = _adjust_channels_order(element=sdata.labels[original_name])

        adata, table_name, table_names = self._get_table_data(sdata, original_name)

        self.viewer.add_labels(
            rgb_labels,
            name=key,
            affine=affine,
            metadata={
                "sdata": sdata,
                "adata": adata,
                "region_key": sdata[table_name].uns["spatialdata_attrs"]["region_key"] if table_name else None,
                "instance_key": sdata[table_name].uns["spatialdata_attrs"]["instance_key"] if table_name else None,
                "table_names": table_names if table_name else None,
                "name": original_name,
                "_active_in_cs": {selected_cs},
                "_current_cs": selected_cs,
                "indices": indices,
            },
        )

    def add_sdata_points(self, sdata: SpatialData, key: str, selected_cs: str, multi: bool) -> None:
        """
        Add a points element in a spatial data object to the viewer.

        Parameters
        ----------
        sdata
            The spatial data object containing the points element.
        key
            The name of the points element in the spatialdata object.
        selected_cs
            The coordinate system in which the points layer is to be loaded.
        multi
            Whether there are multiple spatialdata objects present in the viewer.
        """
        original_name = key
        if multi:
            original_name = original_name[: original_name.rfind("_")]

        points = sdata.points[original_name].compute()
        affine = _get_transform(sdata.points[original_name], selected_cs)
        adata, table_name, table_names = self._get_table_data(sdata, original_name)

        if len(points) < config.POINT_THRESHOLD:
            subsample = None
        else:
            logger.info("Subsampling points because the number of points exceeds the currently supported 100 000.")
            gen = np.random.default_rng()
            subsample = np.sort(gen.choice(len(points), size=config.POINT_THRESHOLD, replace=False))  # same as indices

        subsample_points = points.iloc[subsample] if subsample is not None else points
        if subsample is not None and table_name is not None:
            _, adata = _left_join_spatialelement_table(
                {"points": {original_name: subsample_points}}, sdata[table_name], match_rows="left"
            )
        xy = subsample_points[["y", "x"]].values
        np.fliplr(xy)
        # radii_size = _calc_default_radii(self.viewer, sdata, selected_cs)
        radii_size = 3
        layer = self.viewer.add_points(
            xy,
            name=key,
            size=radii_size * 2,
            affine=affine,
            edge_width=0.0,
            metadata={
                "sdata": sdata,
                "adata": adata,
                "name": original_name,
                "region_key": sdata[table_name].uns["spatialdata_attrs"]["region_key"] if table_name else None,
                "instance_key": sdata[table_name].uns["spatialdata_attrs"]["instance_key"] if table_name else None,
                "table_names": table_names if table_name else None,
                "_active_in_cs": {selected_cs},
                "_current_cs": selected_cs,
                "_n_indices": len(points),
                "indices": subsample_points.index.to_list(),
                "_columns_df": (
                    subsample_excl_coords
                    if (subsample_excl_coords := subsample_points.drop(["x", "y"], axis=1)).shape[1] != 0
                    else None
                ),
            },
        )
        assert affine is not None
        self._adjust_radii_of_points_layer(layer=layer, affine=affine)

    def _adjust_radii_of_points_layer(self, layer: Layer, affine: npt.ArrayLike) -> None:
        """When visualizing circles as points, we need to adjust the radii manually after an affine transformation."""
        assert isinstance(affine, np.ndarray)

        metadata = layer.metadata
        element = metadata["sdata"][metadata["name"]]
        # we don't adjust the radii of dask dataframes (points) since there was no radius to start with (we use an
        # heuristic to calculate the radius in _calc_default_radii())
        if isinstance(element, DaskDataFrame):
            return
        radii = element.radius.to_numpy()

        axes: tuple[str, ...]
        if affine.shape == (3, 3):
            axes = ("y", "x")
        elif affine.shape == (4, 4):
            axes = ("z", "y", "x")
        else:
            raise ValueError(f"Invalid affine shape: {affine.shape}")
        affine_transformation = Affine(affine, input_axes=axes, output_axes=axes)

        new_radii = scale_radii(radii=radii, affine=affine_transformation, axes=axes)

        # the points size is the diameter, in "data pixels" of the current coordinate system, so we need to scale by
        # scale factor of the affine transformation. This scale factor is an approximation when the affine
        # transformation is anisotropic.
        matrix = affine_transformation.to_affine_matrix(input_axes=axes, output_axes=axes)
        eigenvalues = np.linalg.eigvals(matrix[:-1, :-1])
        modules = np.absolute(eigenvalues)
        scale_factor = np.mean(modules)

        layer.size = 2 * new_radii / scale_factor

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
                    if layer._type_string == "points":
                        self._adjust_radii_of_points_layer(layer, affine)
