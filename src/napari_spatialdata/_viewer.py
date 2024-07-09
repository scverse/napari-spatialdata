from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
import packaging.version
import pandas as pd
from anndata import AnnData
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from loguru import logger
from napari import Viewer
from napari.layers import Image, Labels, Points, Shapes
from napari.utils.notifications import show_info
from qtpy.QtCore import QObject, Signal
from shapely import Polygon
from spatialdata import get_element_annotators, get_element_instances
from spatialdata._core.query.relational_query import _left_join_spatialelement_table
from spatialdata.models import PointsModel, ShapesModel, TableModel, force_2d, get_channels
from spatialdata.transformations import Affine, Identity
from spatialdata.transformations._utils import scale_radii

from napari_spatialdata._model import DataModel
from napari_spatialdata.constants import config
from napari_spatialdata.utils._utils import (
    _adjust_channels_order,
    _get_ellipses_from_circles,
    _get_init_metadata_adata,
    _get_transform,
    _transform_coordinates,
    get_duplicate_element_names,
    get_napari_version,
)
from napari_spatialdata.utils._viewer_utils import _get_polygons_properties

if TYPE_CHECKING:
    import numpy.typing as npt
    from napari.layers import Layer
    from napari.utils.events import Event, EventedList
    from spatialdata import SpatialData


class SpatialDataViewer(QObject):
    layer_saved = Signal(object)
    layer_linked = Signal(object)

    def __init__(self, viewer: Viewer, sdata: EventedList) -> None:
        super().__init__()
        self.viewer = viewer
        self.sdata = sdata
        self._model = DataModel()
        self._layer_event_caches: dict[str, list[dict[str, Any]]] = {}
        self.viewer.bind_key("Shift-L", self._inherit_metadata, overwrite=True)
        self.viewer.bind_key("Shift-E", self._save_to_sdata, overwrite=True)
        self.viewer.layers.events.inserted.connect(self._on_layer_insert)
        self._active_layer_table_names = None
        self.viewer.layers.events.removed.connect(self._on_layer_removed)

        # Used to check old layer name. This because event emitted does not contain this information.
        self.layer_names: set[str | None] = set()

    @property
    def model(self) -> DataModel:
        return self._model

    def _on_layer_insert(self, event: Event) -> None:
        layer = event.value
        if layer.metadata.get("sdata"):
            self.layer_names.add(layer.name)
            self._layer_event_caches[layer.name] = []
            layer.events.data.connect(self._update_cache_indices)
            layer.events.name.connect(self._validate_name)
        else:
            if any(layer.name in sdata for sdata in self.sdata):
                layer.name = layer.name + "_external"

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
        if event.action == "remove" or (type(event.source) is not Points and event.action == "change"):
            # We overwrite the indices so they correspond to indices in the dataframe
            napari_indices = sorted(event.data_indices, reverse=True)
            event.indices = tuple(event.source.metadata["indices"][i] for i in napari_indices)
            if event.action == "remove":
                for i in napari_indices:
                    del event.source.metadata["indices"][i]
        elif type(event.source) is Points and event.action == "change":
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

    def _get_spatial_element_name(self, layer: Layer, spatial_element_name: str | None) -> str:
        if spatial_element_name is None:
            spatial_element_name = layer.name
        else:
            layer.name = spatial_element_name
        return str(spatial_element_name)

    def _delete_from_disk(self, sdata: SpatialData, element_name: str, overwrite: bool) -> None:
        if element_name in sdata and len(sdata.locate_element(sdata[element_name])) != 0:
            if overwrite:
                sdata.delete_element_from_disk(element_name)
            else:
                raise OSError(f"`{element_name}` already exists. Use overwrite=True to rewrite.")

    def _write_element_to_disk(
        self,
        sdata: SpatialData,
        element_name: str,
        element: tuple[DaskDataFrame | GeoDataFrame | AnnData],
        overwrite: bool,
    ) -> None:
        if sdata.is_backed:
            self._delete_from_disk(sdata, element_name, overwrite)
            sdata[element_name] = element
            sdata.write_element(element_name)
        else:
            sdata[element_name] = element
            logger.warning("Spatialdata object is not stored on disk, could only add element in memory.")

    def _save_points_to_sdata(
        self, layer_to_save: Points, spatial_element_name: str | None, overwrite: bool
    ) -> tuple[DaskDataFrame, str]:
        sdata = layer_to_save.metadata["sdata"]
        coordinate_system = layer_to_save.metadata["_current_cs"]
        transformation = {coordinate_system: Identity()}

        spatial_element_name = self._get_spatial_element_name(layer_to_save, spatial_element_name)

        if len(layer_to_save.data) == 0:
            raise ValueError("Cannot export a points element with no points")
        transformed_data = np.array([layer_to_save.data_to_world(xy) for xy in layer_to_save.data])
        swap_data = np.fliplr(transformed_data)
        # ignore z axis if present
        if swap_data.shape[1] == 3:
            swap_data = swap_data[:, :2]
        parsed = PointsModel.parse(swap_data, transformations=transformation)

        self._write_element_to_disk(sdata, spatial_element_name, parsed, overwrite)

        return parsed, coordinate_system

    def _save_table_to_sdata(
        self,
        layer_to_save: Layer,
        table_name: str,
        spatial_element_name: str | None,
        table_columns: list[str] | None,
        overwrite: bool,
    ) -> None:
        sdata = layer_to_save.metadata["sdata"]
        feature_color_column = [column for column in layer_to_save.features.columns if "color" in column]
        if len(feature_color_column) > 0:
            color_column_name = feature_color_column[0]
            class_column_name = color_column_name.split("_")[0]
            region_key = "region" if not layer_to_save.metadata["region_key"] else layer_to_save.metadata["region_key"]
            instance_key = (
                "instance_id" if not layer_to_save.metadata["instance_key"] else layer_to_save.metadata["instance_key"]
            )

            copy_table = layer_to_save.features.copy()
            if all(row == "" for row in copy_table["description"]):
                copy_table.drop(columns=["description"], inplace=True)
            if len(categories := copy_table["annotator"].cat.categories) == 1 and categories[0] == "":
                copy_table.drop(columns=["annotator"], inplace=True)
            class_to_color_mapping = copy_table.set_index(class_column_name)[color_column_name].to_dict()
            copy_table.drop(columns=[color_column_name], inplace=True)

            copy_table.reset_index(names="instance_id", inplace=True)

            if spatial_element_name not in copy_table["region"].cat.categories:
                layer_to_save.name = spatial_element_name
                copy_table["region"] = spatial_element_name
            if table_columns:
                color_column_name = table_columns[1]
                copy_table.rename(columns={"class": table_columns[0]}, inplace=True)
            copy_table = AnnData(obs=copy_table, uns={color_column_name: class_to_color_mapping})

            sdata_table = TableModel.parse(
                copy_table,
                region=layer_to_save.name,
                region_key=region_key,
                instance_key=instance_key,
            )

            self._write_element_to_disk(sdata, table_name, sdata_table, overwrite)

    def _save_shapes_to_sdata(
        self, layer_to_save: Shapes, spatial_element_name: str | None, overwrite: bool
    ) -> tuple[GeoDataFrame, str]:
        sdata = layer_to_save.metadata["sdata"]
        coordinate_system = layer_to_save.metadata["_current_cs"]
        transformation = {coordinate_system: Identity()}
        spatial_element_name = self._get_spatial_element_name(layer_to_save, spatial_element_name)

        if len(layer_to_save.data) == 0:
            raise ValueError("Cannot export a shapes element with no shapes")

        polygons: list[Polygon] = [Polygon(i) for i in _transform_coordinates(layer_to_save.data, f=lambda x: x[::-1])]
        gdf = GeoDataFrame({"geometry": polygons})

        force_2d(gdf)
        parsed = ShapesModel.parse(gdf, transformations=transformation)

        self._write_element_to_disk(sdata, spatial_element_name, parsed, overwrite)

        return parsed, coordinate_system

    def save_to_sdata(
        self,
        layers: list[Layer] | None = None,
        spatial_element_name: str | None = None,
        table_name: str | None = None,
        table_columns: list[str] | None = None,
        overwrite: bool = False,
    ) -> None:
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
        selected_layers = layers if layers else self.viewer.layers.selection
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
        if isinstance(selected, Points):
            parsed, cs = self._save_points_to_sdata(selected, spatial_element_name, overwrite)
        elif isinstance(selected, Shapes):
            parsed, cs = self._save_shapes_to_sdata(selected, spatial_element_name, overwrite)
            if table_name:
                self._save_table_to_sdata(selected, table_name, spatial_element_name, table_columns, overwrite)
        elif isinstance(selected, (Image, Labels)):
            raise NotImplementedError
        else:
            raise ValueError(f"Layer of type {type(selected)} cannot be saved.")

        self.layer_names.add(selected.name)
        self._layer_event_caches[selected.name] = []
        self._update_metadata(selected, parsed)
        selected.events.data.connect(self._update_cache_indices)
        selected.events.name.connect(self._validate_name)
        self.layer_saved.emit(cs)
        show_info("Layer saved")

    def _update_metadata(self, layer: Layer, model: DaskDataFrame | None) -> None:
        layer.metadata["name"] = layer.name
        layer.metadata["_n_indices"] = len(layer.data)
        layer.metadata["indices"] = list(i for i in range(len(layer.data)))  # noqa: C400

        sdata = layer.metadata["sdata"]
        adata, table_name, table_names = self._get_table_data(sdata, layer.metadata["name"])
        if adata is not None:
            layer.metadata["adata"] = adata
            layer.metadata["region_key"] = adata.uns["spatialdata_attrs"]["region_key"] if table_name else None
            layer.metadata["instance_key"] = adata.uns["spatialdata_attrs"]["instance_key"] if table_name else None
            layer.metadata["table_names"] = table_names if table_name else None
            layer.metadata["_columns_df"] = None

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
            self.layer_linked.emit(layer)

        show_info(f"Layer(s) inherited info from {ref_layer}")

    def _get_table_data(
        self, sdata: SpatialData, element_name: str
    ) -> tuple[AnnData | None, str | None, list[str | None]]:
        table_names = list(get_element_annotators(sdata, element_name))
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

        channels = get_channels(sdata.images[original_name])
        adata = AnnData(shape=(0, len(channels)), var=pd.DataFrame(index=channels))

        # TODO: type check
        self.viewer.add_image(
            rgb_image,
            rgb=rgb,
            name=key,
            affine=affine,
            metadata={
                "adata": adata,
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
        version = get_napari_version()
        kwargs: dict[str, Any] = (
            {"edge_width": 0.0} if version <= packaging.version.parse("0.4.20") else {"border_width": 0.0}
        )
        if CIRCLES_AS_POINTS:
            layer = self.viewer.add_points(
                yx,
                name=key,
                affine=affine,
                size=1,  # the sise doesn't matter here since it will be adjusted in _adjust_radii_of_points_layer
                metadata=metadata,
                **kwargs,
            )
            assert affine is not None
            self._adjust_radii_of_points_layer(layer=layer, affine=affine)
        else:
            if version <= packaging.version.parse("0.4.20"):
                kwargs |= {"edge_color": "white"}
            else:
                kwargs |= {"border_color": "white"}
            # useful code to have readily available to debug the correct radius of circles when represented as points
            ellipses = _get_ellipses_from_circles(yx=yx, radii=radii)
            self.viewer.add_shapes(
                ellipses,
                shape_type="ellipse",
                name=key,
                face_color="white",
                affine=affine,
                metadata=metadata,
                **kwargs,
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

        indices = get_element_instances(sdata.labels[original_name])
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
            logger.info(
                f"Subsampling points because the number of points exceeds the currently supported "
                f"{config.POINT_THRESHOLD}. You can change this threshold with "
                f"```from napari_spatialdata.constants import config\n"
                f"config.POINT_THRESHOLD = <new_threshold>```"
            )
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
        version = get_napari_version()
        kwargs = {"edge_width": 0.0} if version <= packaging.version.parse("0.4.20") else {"border_width": 0.0}
        layer = self.viewer.add_points(
            xy,
            name=key,
            size=radii_size * 2,
            affine=affine,
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
            **kwargs,
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
