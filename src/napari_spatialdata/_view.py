from typing import Any, Optional, Sequence, FrozenSet, Dict

from loguru import logger
from anndata import AnnData
from magicgui import magicgui
from napari.layers import Layer, Labels
from napari.viewer import Viewer
from qtpy.QtWidgets import QLabel, QWidget, QComboBox, QVBoxLayout
from geopandas import GeoDataFrame
from shapely.geometry import Polygon
import numpy as np
import napari
import pandas as pd

from napari_spatialdata._model import ImageModel
from napari_spatialdata._utils import (
    NDArrayA,
    _get_categorical,
    _points_inside_triangles,
)
from napari_spatialdata._widgets import (
    CBarWidget,
    AListWidget,
    ObsmIndexWidget,
    RangeSliderWidget,
    CoordinateSystemSelector,
)
from napari_spatialdata._constants._pkg_constants import Key
from napari.utils.notifications import show_info

__all__ = ["QtAdataViewWidget"]


class QtAdataViewWidget(QWidget):
    """Adata viewer widget."""

    def __init__(self, viewer: Viewer):
        super().__init__()

        self._viewer = viewer
        self._model = ImageModel()
        self._model.palette = "tab20"

        self._coordinate_system_selector = CoordinateSystemSelector(self._viewer, multiselect=False)
        self._coordinate_system_selector.setCurrentRow(0)

        # coordiante systyem
        self.setLayout(QVBoxLayout())
        cs_label = QLabel("Coordinate space:")
        self.layout().addWidget(cs_label)
        self.layout().addWidget(self._coordinate_system_selector)

        # this is to select which layer to use (only layers with anndata tables are listed)
        # below there is another layer selection widget to select which layer to use (within the adata table)
        # self._layer_selection_widget = magicgui(
        #     self._select_layer,
        #     layer={"choices": self._get_layer},
        #     auto_call=True,
        #     call_button=False,
        # )
        # self._layer_selection_widget()
        # self.layout().addWidget(self._layer_selection_widget.native)
        self._layer_selection_widget = QComboBox()
        layers = self._get_adata_layers()
        layer_names = [layer.name for layer in layers]
        self._layer_selection_widget.addItems(layer_names)
        self.layout().addWidget(self._layer_selection_widget)
        self._viewer.layers.selection.events.changed.connect(self._layer_selection_changed)
        self._layer_selection_widget.currentTextChanged.connect(self._select_layer)
        self._layer_selection_widget.currentTextChanged.emit(self._layer_selection_widget.currentText())
        self._viewer.layers.events.connect(self._update_visibility)
        # self._viewer.layers.vis

        # obs
        obs_label = QLabel("Observations:")
        obs_label.setToolTip("Keys in `adata.obs` containing cell observations.")
        self.obs_widget = AListWidget(self.viewer, self.model, attr="obs")
        self.layout().addWidget(obs_label)
        self.layout().addWidget(self.obs_widget)
        self._coordinate_system_selector.currentTextChanged.connect(self.obs_widget._coordinateSystemChanged)
        self.obs_widget._current_coordinate_system = self._coordinate_system_selector._current_coordinate_system

        # gene
        var_label = QLabel("Genes:")
        var_label.setToolTip("Gene names from `adata.var_names` or `adata.raw.var_names`.")
        self.var_widget = AListWidget(self.viewer, self.model, attr="var")
        self.var_widget.setAdataLayer("X")
        self._coordinate_system_selector.currentTextChanged.connect(self.var_widget._coordinateSystemChanged)
        self.var_widget._current_coordinate_system = self._coordinate_system_selector._current_coordinate_system

        # layers
        # luca: what is this for? we alredy have a layer selection above (which is the one to choose which adata table to use)
        adata_layer_label = QLabel("Layers:")
        adata_layer_label.setToolTip("Keys in `adata.layers` used when visualizing gene expression.")
        self.adata_layer_widget = QComboBox()
        self.adata_layer_widget.addItem("X", None)
        self.adata_layer_widget.addItems(self._get_adata_layer())
        self.adata_layer_widget.currentTextChanged.connect(self.var_widget.setAdataLayer)

        self.layout().addWidget(self.adata_layer_widget)
        self.layout().addWidget(var_label)
        self.layout().addWidget(self.var_widget)

        # obsm
        obsm_label = QLabel("Obsm:")
        obsm_label.setToolTip("Keys in `adata.obsm` containing multidimensional cell information.")
        self.obsm_widget = AListWidget(self.viewer, self.model, attr="obsm", multiselect=False)
        self.obsm_index_widget = ObsmIndexWidget(self.model)
        self.obsm_index_widget.setToolTip("Indices for current key in `adata.obsm`.")
        self.obsm_index_widget.currentTextChanged.connect(self.obsm_widget.setIndex)
        self.obsm_widget.itemClicked.connect(self.obsm_index_widget.addItems)
        self._coordinate_system_selector.currentTextChanged.connect(self.obsm_widget._coordinateSystemChanged)
        self.obsm_widget._current_coordinate_system = self._coordinate_system_selector._current_coordinate_system

        self.layout().addWidget(obsm_label)
        self.layout().addWidget(self.obsm_widget)
        self.layout().addWidget(self.obsm_index_widget)

        # gene
        var_points = QLabel("Points:")
        var_points.setToolTip("Gene names from points.")
        self.var_points_widget = AListWidget(self.viewer, self.model, attr="points")
        self._coordinate_system_selector.currentTextChanged.connect(self.var_points_widget._coordinateSystemChanged)
        self.var_points_widget._current_coordinate_system = self._coordinate_system_selector._current_coordinate_system

        self.layout().addWidget(var_points)
        self.layout().addWidget(self.var_points_widget)

        # scalebar
        colorbar = CBarWidget()
        self.slider = RangeSliderWidget(self.viewer, self.model, colorbar=colorbar)
        self._viewer.window.add_dock_widget(self.slider, area="left", name="slider")
        self._viewer.window.add_dock_widget(colorbar, area="left", name="colorbar")
        self.viewer.layers.selection.events.active.connect(self.slider._onLayerChange)

        self.viewer.bind_key("Shift-E", self.export)
        self.model.events.adata.connect(self._on_layer_update)

    def _on_layer_update(self, event: Optional[Any] = None) -> None:
        """When the model updates the selected layer, update the relevant widgets."""
        logger.info("Updating layer.")

        self.adata_layer_widget.clear()
        self.adata_layer_widget.addItem("X", None)
        self.adata_layer_widget.addItems(self._get_adata_layer())
        self.obs_widget._onChange()
        self.var_widget._onChange()
        self.obsm_widget._onChange()
        self.var_points_widget._onChange()

    def _update_visibility(self, event: Optional[Any] = None) -> None:
        """
        When the visibility of a layer changes, update the relevant widgets.
        """
        if event.type == "visible":
            self._layer_selection_changed(None)

    def _layer_selection_changed(self, event: Optional[Any]) -> None:
        """
        When the napari layer selection or visibily changes, update the combobox for selecting napari layer with anndata.
        """
        ##
        all_layers_with_adata = []
        all_layers_with_adata_visible = []
        for layer in self.viewer.layers:
            if "adata" in layer.metadata and layer.metadata["adata"] is not None:
                all_layers_with_adata.append(layer.name)
                if layer.visible:
                    all_layers_with_adata_visible.append(layer.name)
        ##
        all_layers_selected = [a.name for a in self.viewer.layers.selection]
        all_layers_with_adata_selected_and_visible = [
            layer for layer in all_layers_selected if layer in all_layers_with_adata_visible
        ]
        ##
        old_current_text = self._layer_selection_widget.currentText()
        all_items = [self._layer_selection_widget.itemText(i) for i in range(self._layer_selection_widget.count())]
        if set(all_items) != set(all_layers_with_adata):
            # update the values of the combobox
            self._layer_selection_widget.clear()
            self._layer_selection_widget.addItems(all_layers_with_adata)
            if old_current_text in all_layers_with_adata:
                self._layer_selection_widget.blockSignals(True)
                self._layer_selection_widget.setCurrentText(old_current_text)
                self._layer_selection_widget.blockSignals(False)
        ##
        if (
            old_current_text not in all_layers_with_adata_selected_and_visible
            and len(all_layers_with_adata_selected_and_visible) > 0
        ):
            first_selected = all_layers_with_adata_selected_and_visible[0]
            self._layer_selection_widget.setCurrentText(first_selected)
        elif old_current_text not in all_layers_with_adata_visible and len(all_layers_with_adata_visible) > 0:
            first_selected = all_layers_with_adata_visible[0]
            self._layer_selection_widget.setCurrentText(first_selected)

    def _get_layer_by_name(self, layer_name: str) -> Optional[Layer]:
        """Get the layer by name."""
        for layer in self.viewer.layers:
            if layer.name == layer_name:
                return layer
        return None

    def _select_layer(self, layer_name: str) -> None:
        """Napari layers."""
        layer = self._get_layer_by_name(layer_name)
        if layer is None:
            # this happens when there is no layer with adata to show
            return
        if "adata" not in layer.metadata:
            return
        self.model.layer = layer
        # if layer is not None and "adata" in layer.metadata:
        self.model.adata = layer.metadata["adata"]
        self.model.library_id = layer.metadata["library_id"]
        # TODO: replace with transformations
        # self.model.scale = self.model.adata.uns[Key.uns.spatial][self.model.library_id][Key.uns.scalefactor_key][
        #     self.model.scale_key
        # ]
        self.model.scale = 1.0
        self.model.coordinates = np.insert(
            self.model.adata.obsm[Key.obsm.spatial][:, ::-1][:, :2] * self.model.scale, 0, values=0, axis=1
        )
        if "points" in layer.metadata:
            self.model.points_coordinates = layer.metadata["points"].X
            self.model.points_var = layer.metadata["points"].obs["gene"]
            self.model.point_diameter = np.array([0.0] + [layer.metadata["point_diameter"]] * 2) * self.model.scale
        # workaround to support different sizes for different point, for layers coming from SpatialData
        if "region_radius" in self.model.adata.obsm:
            self.model.spot_diameter = 2 * self.model.adata.obsm["region_radius"]
        else:
            # TODO: replace
            # self.model.spot_diameter = (
            #     np.array([0.0] + [Key.uns.spot_diameter(self.model.adata, Key.obsm.spatial, self.model.library_id)] * 2)
            #     * self.model.scale
            # )
            self.model.spot_diameter = 1.0
        self.model.labels_key = layer.metadata["labels_key"] if isinstance(layer, Labels) else None

    def _get_adata_layers(self) -> Sequence[Optional[str]]:
        adata_layers = []
        for layer in self._viewer.layers:
            if isinstance(layer.metadata.get("adata", None), AnnData):
                adata_layers.append(layer)
        # if not len(adata_layers):
        #     raise NotImplementedError(
        #         "`AnnData` not found in any `layer.metadata`. This plugin requires `AnnData` in at least one layer."
        #     )
        return adata_layers

    def _get_adata_layer(self) -> Sequence[Optional[str]]:
        adata = self.model.adata
        if adata is not None:
            adata_layers = list(self.model.adata.layers.keys())
            if len(adata_layers):
                return adata_layers
        return [None]

    def export(self, _: napari.viewer.Viewer) -> None:
        """Export shapes into :class:`AnnData` object."""
        selection = self.viewer.layers.selection
        if len(selection) != 1:
            show_info("Cannot save. Please select only one layer to export points or polygons.")
            return
        layer = selection.__iter__().__next__()
        ##
        sdatas = []
        for ll in self.viewer.layers:
            if ll.visible:
                if "sdata" in ll.metadata:
                    sdata = ll.metadata["sdata"]
                    sdatas.append(sdata)
        sdata_ids = set([id(sdata) for sdata in sdatas])
        if len(sdata_ids) == 0:
            raise RuntimeError(
                "Cannot save polygons because no layer associated with a SpatialData object is " "currently visible."
            )
        elif len(sdata_ids) > 1:
            logger.warning("More than one SpatialData object is currently visible. Saving polygons for the first one.")
        ##

        sdata = sdatas[0]
        from spatialdata._core.transformations import Identity
        from spatialdata._core.models import ShapesModel, PolygonsModel

        # get current coordinate system
        selected = self._coordinate_system_selector.selectedItems()

        assert len(selected) >= 1
        if len(selected) > 1:
            logger.warning("More than one coordinate system selected. Using the first one to save the annotations.")
        cs = selected[0].text()
        assert cs in sdata.coordinate_systems

        key = f"{layer.name}"
        zarr_name = key.replace(" ", "_").replace("[", "").replace("]", "").replace(":", "_")

        # TODO: polygons is using data_to_world(), points is doing this manually by applying .affine; these two ways are equivalent but the first is quicker; use just it
        if isinstance(layer, napari.layers.points.points.Points):
            coords = layer.data
            raw_sizes = layer._size
            sizes = []
            for i, row in enumerate(raw_sizes):
                assert len(set(row)) == 1
                size = row[0]
                sizes.append(size)
            sizes_array = np.array(sizes)
            # we apply the transformation that is used in the layer
            assert len(coords.shape) == 2
            p = np.vstack([coords.T, np.ones(coords.shape[0])])
            q = layer.affine.affine_matrix @ p
            coords = q[: coords.shape[-1], :].T

            # TODO: deal with the 3D case
            if layer.ndim == 3:
                coords = coords[:, 1:]
            else:
                assert layer.ndim == 2

            # coords from napari are in the yx coordinate systems, we want to store them as xy
            coords = np.fliplr(coords)
            shapes = ShapesModel.parse(
                coords=coords, transformations={cs: Identity()}, shape_type="Circle", shape_size=sizes_array
            )
            # sequence.transform(shapes).obsm['spatial']
            sdata.add_shapes(name=zarr_name, shapes=shapes, overwrite=True)
            show_info(f"Shapes saved in the SpatialData object")
        elif isinstance(layer, napari.layers.shapes.shapes.Shapes):
            coords = [np.array([layer.data_to_world(xy) for xy in shape._data]) for shape in layer._data_view.shapes]
            # TODO: deal with the 3D case (is there a 3D case?)
            if layer.ndim == 3:
                coords = [c[:, 1:] for c in coords]
            else:
                assert layer.ndim == 2
            polygons = []
            for polygon_coords in coords:
                # coords from napari are in the yx coordinate systems, we want to store them as xy
                polygon_coords = np.fliplr(polygon_coords)
                polygon = Polygon(polygon_coords)
                polygons.append(polygon)
            gdf = GeoDataFrame({"geometry": polygons})
            parsed = PolygonsModel.parse(gdf)
            sdata.add_polygons(name=zarr_name, polygons=parsed, overwrite=True)
            show_info(f"Polygons saved in the SpatialData object")
        else:
            show_info("You can only save a layer of type points or polygons.")

    def _save_shapes(self, layer: napari.layers.Shapes, key: str) -> None:
        shape_list = layer._data_view
        triangles = shape_list._mesh.vertices[shape_list._mesh.displayed_triangles]

        # TODO(giovp): check if view and save accordingly
        points_mask: NDArrayA = _points_inside_triangles(self.model.coordinates[:, 1:], triangles)

        logger.info("Saving layer shapes.")

        self._model._adata.obs[key] = pd.Categorical(points_mask)
        self._model._adata.uns[key] = {"meshes": layer.data.copy()}

    def _update_obs_items(self, key: str) -> None:
        self.obs_widget.addItems(key)
        if key in self.layernames:
            # update already present layer
            layer = self.viewer.layers[key]
            layer.face_color = _get_categorical(self.model.adata, key)
            layer._update_thumbnail()
            layer.refresh_colors()

    @property
    def viewer(self) -> napari.Viewer:
        """:mod:`napari` viewer."""
        return self._viewer

    @property
    def model(self) -> ImageModel:
        """:mod:`napari` viewer."""
        return self._model

    @property
    def layernames(self) -> FrozenSet[str]:
        """Names of :attr:`napari.Viewer.layers`."""
        return frozenset(layer.name for layer in self.viewer.layers)
