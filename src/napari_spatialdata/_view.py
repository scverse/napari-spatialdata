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
            auto_call=True,
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

        # gene
        var_points = QLabel("Points:")
        var_points.setToolTip("Gene names from points.")
        self.var_points_widget = AListWidget(self.viewer, self.model, attr="points")

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
        self.adata_layer_widget.clear()
        self.adata_layer_widget.addItem("X", None)
        self.adata_layer_widget.addItems(self._get_adata_layer())
        self.obs_widget._onChange()
        self.var_widget._onChange()
        self.obsm_widget._onChange()
        self.var_points_widget._onChange()

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
        if "points" in layer.metadata:
            self.model.points_coordinates = layer.metadata["points"].X
            self.model.points_var = layer.metadata["points"].obs["gene"]
            self.model.point_diameter = np.array([0.0] + [layer.metadata["point_diameter"]] * 2) * self.model.scale
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
