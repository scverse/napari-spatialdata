from typing import Any, Optional, Sequence, FrozenSet

from scanpy import logging as logg
from anndata import AnnData
from magicgui import magicgui
from napari.layers import Layer, Labels
from napari.viewer import Viewer
from qtpy.QtWidgets import QLabel, QWidget, QComboBox, QVBoxLayout
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
)
from napari_spatialdata._constants._pkg_constants import Key

__all__ = ["QtAdataViewWidget"]


class QtAdataViewWidget(QWidget):
    """Adata viewer widget."""

    def __init__(self, viewer: Viewer):
        super().__init__()

        self._viewer = viewer
        self._model = ImageModel()

        self._layer_selection_widget = magicgui(
            self._select_layer,
            layer={"choices": self._get_layer},
            auto_call=False,
            call_button=False,
        )
        self._layer_selection_widget()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self._layer_selection_widget.native)

        # obs
        obs_label = QLabel("Observations:")
        obs_label.setToolTip("Keys in `adata.obs` containing cell observations.")
        self.obs_widget = AListWidget(self.viewer, self.model, attr="obs")
        self.layout().addWidget(obs_label)
        self.layout().addWidget(self.obs_widget)

        # gene
        var_label = QLabel("Genes:")
        var_label.setToolTip("Gene names from `adata.var_names` or `adata.raw.var_names`.")
        self.var_widget = AListWidget(self.viewer, self.model, attr="var")
        self.var_widget.setAdataLayer("X")

        # layers
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

        self.layout().addWidget(obsm_label)
        self.layout().addWidget(self.obsm_widget)
        self.layout().addWidget(self.obsm_index_widget)

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
        self.adata_layer_widget.clear()
        self.adata_layer_widget.addItem("X", None)
        self.adata_layer_widget.addItems(self._get_adata_layer())
        self.obs_widget._onChange()
        self.var_widget._onChange()
        self.obsm_widget._onChange()

    def _select_layer(self, layer: Layer) -> None:
        """Napari layers."""
        self.model.layer = layer
        # if layer is not None and "adata" in layer.metadata:
        self.model.adata = layer.metadata["adata"]
        self.model.library_id = layer.metadata["library_id"]
        self.model.scale = self.model.adata.uns[Key.uns.spatial][self.model.library_id][Key.uns.scalefactor_key][
            self.model.scale_key
        ]
        self.model.coordinates = np.insert(
            self.model.adata.obsm[Key.obsm.spatial][:, ::-1][:, :2] * self.model.scale, 0, values=0, axis=1
        )
        self.model.spot_diameter = (
            np.array([0.0] + [Key.uns.spot_diameter(self.model.adata, Key.obsm.spatial, self.model.library_id)] * 2)
            * self.model.scale
        )
        self.model.labels_key = layer.metadata["labels_key"] if isinstance(layer, Labels) else None

    def _get_layer(self, combo_widget: QComboBox) -> Sequence[Optional[str]]:
        adata_layers = []
        for layer in self._viewer.layers:
            if isinstance(layer.metadata.get("adata", None), AnnData):
                adata_layers.append(layer)
        if not len(adata_layers):
            raise NotImplementedError(
                "`AnnData` not found in any `layer.metadata`. This plugin requires `AnnData` in at least one layer."
            )
        return adata_layers

    def _get_adata_layer(self) -> Sequence[Optional[str]]:
        adata_layers = list(self.model.adata.layers.keys())
        if len(adata_layers):
            return adata_layers
        return [None]

    def export(self, _: napari.viewer.Viewer) -> None:
        """Export shapes into :class:`AnnData` object."""
        for layer in self.viewer.layers:
            if not isinstance(layer, napari.layers.Shapes) or layer not in self.viewer.layers.selection:
                continue
            if not len(layer.data):
                logg.warning(f"Shape layer `{layer.name}` has no visible shapes")
                continue

            key = f"{layer.name}_{self.model.layer.name}"

            logg.info(f"Adding `adata.obs[{key!r}]`\n       `adata.uns[{key!r}]['mesh']`")
            self._save_shapes(layer, key=key)
            self._update_obs_items(key)

    def _save_shapes(self, layer: napari.layers.Shapes, key: str) -> None:
        shape_list = layer._data_view
        triangles = shape_list._mesh.vertices[shape_list._mesh.displayed_triangles]

        # TODO(giovp): check if view and save accordingly
        points_mask: NDArrayA = _points_inside_triangles(self.model.coordinates[:, 1:], triangles)

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


# class ImageView:
#     """
#     View class which initializes :class:`napari.Viewer`.

#     Parameters
#     ----------
#     model
#         Model for this view.
#     controller
#         Controller for this view.
#     """

#     def __init__(self, model: ImageModel, controller: ImageController_):
#         self._model = model
#         self._controller = controller

#     def _init_UI(self) -> None:
#         def update_library(event: Any) -> None:
#             value = tuple(event.value)
#             if len(value) == 3:
#                 lid = value[0]
#             elif len(value) == 4:
#                 lid = value[1]
#             else:
#                 logg.error(f"Unable to set library id from `{value}`")
#                 return

#             self.model.alayer.library_id = lid
#             library_id.setText(f"{self.model.alayer.library_id}")

#         self._viewer = napari.Viewer(title="Squidpy", show=False)
#         self.viewer.bind_key("Shift-E", self.controller.export)
#         parent = self.viewer.window._qt_window

#         # image
#         image_lab = QLabel("Images:")
#         image_lab.setToolTip("Keys in `ImageContainer` containing the image data.")
#         image_widget = LibraryListWidget(self.controller, multiselect=False, unique=True)
#         image_widget.setMaximumHeight(100)
#         image_widget.addItems(tuple(self.model.container))
#         image_widget.setCurrentItem(image_widget.item(0))

#         # gene
#         var_lab = QLabel("Genes:", parent=parent)
#         var_lab.setToolTip("Gene names from `adata.var_names` or `adata.raw.var_names`.")
#         var_widget = AListWidget(self.controller, self.model.alayer, attr="var", parent=parent)

#         # obs
#         obs_label = QLabel("Observations:", parent=parent)
#         obs_label.setToolTip("Keys in `adata.obs` containing cell observations.")
#         self._obs_widget = AListWidget(self.controller, self.model.alayer, attr="obs", parent=parent)

#         # obsm
#         obsm_label = QLabel("Obsm:", parent=parent)
#         obsm_label.setToolTip("Keys in `adata.obsm` containing multidimensional cell information.")
#         obsm_widget = AListWidget(self.controller, self.model.alayer, attr="obsm", multiselect=False, parent=parent)
#         obsm_index_widget = ObsmIndexWidget(self.model.alayer, parent=parent)
#         obsm_index_widget.setToolTip("Indices for current key in `adata.obsm`.")
#         obsm_index_widget.currentTextChanged.connect(obsm_widget.setIndex)
#         obsm_widget.itemClicked.connect(obsm_index_widget.addItems)

#         # layer selection
#         layer_label = QLabel("Layers:", parent=parent)
#         layer_label.setToolTip("Keys in `adata.layers` used when visualizing gene expression.")
#         layer_widget = QComboBox(parent=parent)
#         layer_widget.addItem("X", None)
#         layer_widget.addItems(self.model.adata.layers.keys())
#         layer_widget.currentTextChanged.connect(var_widget.setLayer)
#         layer_widget.setCurrentText("X")

#         # raw selection
#         raw_cbox = TwoStateCheckBox(parent=parent)
#         raw_cbox.setDisabled(self.model.adata.raw is None)
#         raw_cbox.checkChanged.connect(layer_widget.setDisabled)
#         raw_cbox.checkChanged.connect(var_widget.setRaw)
#         raw_layout = QHBoxLayout()
#         raw_label = QLabel("Raw:", parent=parent)
#         raw_label.setToolTip("Whether to access `adata.raw.X` or `adata.X` when visualizing gene expression.")
#         raw_layout.addWidget(raw_label)
#         raw_layout.addWidget(raw_cbox)
#         raw_layout.addStretch()
#         raw_widget = QWidget(parent=parent)
#         raw_widget.setLayout(raw_layout)

#         library_id = QLabel(f"{self.model.alayer.library_id}")
#         library_id.setToolTip("Currently selected library id.")

#         widgets = (
#             image_lab,
#             image_widget,
#             layer_label,
#             layer_widget,
#             raw_widget,
#             var_lab,
#             var_widget,
#             obs_label,
#             self._obs_widget,  # needed for controller to add mask
#             obsm_label,
#             obsm_widget,
#             obsm_index_widget,
#             library_id,
#         )
#         self._colorbar = CBarWidget(self.model.cmap, parent=parent)

#         self.viewer.window.add_dock_widget(self._colorbar, area="left", name="percentile")
#         self.viewer.window.add_dock_widget(widgets, area="right", name="genes")
#         self.viewer.layers.selection.events.changed.connect(self._move_layer_to_front)
#         self.viewer.layers.selection.events.changed.connect(self._adjust_colorbar)
#         self.viewer.dims.events.current_step.connect(update_library)
#         # TODO: find callback that that shows all Z-dimensions and change lib. id to 'All'

#     def _move_layer_to_front(self, event: Any) -> None:
#         try:
#             layer = next(iter(event.added))
#         except StopIteration:
#             return
#         if not layer.visible:
#             return
#         try:
#             index = self.viewer.layers.index(layer)
#         except ValueError:
#             return

#         self.viewer.layers.move(index, -1)

#     def _adjust_colorbar(self, event: Any) -> None:
#         try:
#             layer = next(layer for layer in event.added if isinstance(layer, Points))
#         except StopIteration:
#             return

#         try:
#             self._colorbar.setOclim(layer.metadata["minmax"])
#             self._colorbar.setClim((np.min(layer.properties["value"]), np.max(layer.properties["value"])))
#             self._colorbar.update_color()
#         except KeyError:  # categorical
#             pass

#     @property
#     def layers(self) -> napari.components.layerlist.LayerList:
#         """List of layers of :attr:`napari.Viewer.layers`."""
#         return self.viewer.layers

#     @property
#     def layernames(self) -> frozenset[str]:
#         """Names of :attr:`napari.Viewer.layers`."""
#         return frozenset(layer.name for layer in self.layers)

#     @property
#     def viewer(self) -> napari.Viewer:
#         """:mod:`napari` viewer."""
#         return self._viewer

#     @property
#     def model(self) -> ImageModel:
#         """Model for this view."""
#         return self._model

#     @property
#     def controller(self) -> ImageController_:
#         """Controller for this view."""
#         return self._controller
