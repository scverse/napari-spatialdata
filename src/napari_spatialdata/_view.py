from typing import Any, FrozenSet, Optional, Sequence

import napari
import numpy as np
import pandas as pd
from anndata import AnnData
from loguru import logger
from napari.layers import Labels
from napari.viewer import Viewer
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from napari_spatialdata._constants._pkg_constants import Key
from napari_spatialdata._model import ImageModel
from napari_spatialdata._scatterwidgets import AxisWidgets, MatplotlibWidget
from napari_spatialdata._utils import (
    NDArrayA,
    _get_categorical,
    _points_inside_triangles,
)
from napari_spatialdata._widgets import (
    AListWidget,
    CBarWidget,
    ComponentWidget,
    RangeSliderWidget,
)

__all__ = ["QtAdataViewWidget", "QtAdataScatterWidget"]


class QtAdataScatterWidget(QWidget):
    """Adata viewer widget."""

    def __init__(self, viewer: Viewer):
        super().__init__()

        self._viewer = viewer
        self._model = ImageModel()

        self._select_layer()
        self._viewer.layers.selection.events.changed.connect(self._select_layer)
        self._viewer.layers.selection.events.changed.connect(self._on_selection)

        self.setLayout(QGridLayout())

        # Matplotlib

        self.matplotlib_widget = MatplotlibWidget(self.viewer, self.model)
        self.layout().addWidget(self.matplotlib_widget, 1, 0, 1, 3)

        self.x_widget = AxisWidgets(self.viewer, self.model, "X-axis")
        self.layout().addWidget(self.x_widget, 2, 0, 6, 1)

        self.y_widget = AxisWidgets(self.viewer, self.model, "Y-axis")
        self.layout().addWidget(self.y_widget, 2, 1, 6, 1)

        self.color_widget = AxisWidgets(self.viewer, self.model, "Color", True)
        self.layout().addWidget(self.color_widget, 2, 2, 6, 1)

        self.plot_button_widget = QPushButton("Plot")
        self.plot_button_widget.clicked.connect(
            lambda: self.matplotlib_widget._onClick(
                self.x_widget.widget.data,
                self.y_widget.widget.data,
                self.color_widget.widget.data,  # type:ignore[arg-type]
                self.x_widget.getFormattedLabel(),
                self.y_widget.getFormattedLabel(),
                self.color_widget.getFormattedLabel(),
            )
        )

        self.export_button_widget = QPushButton("Export")
        self.export_button_widget.clicked.connect(self.export)

        self.layout().addWidget(self.plot_button_widget, 8, 0, 1, 2)
        self.layout().addWidget(self.export_button_widget, 8, 2, 1, 2)

        self.model.events.adata.connect(self._on_selection)

    def export(self) -> None:
        """Export shapes."""
        if (self.matplotlib_widget.selector) is None or (self.matplotlib_widget.selector.exported_data is None):
            raise ValueError("Data points haven't been selected from the matplotlib visualisation.")

        self.matplotlib_widget.selector.export(self.model.adata)

    def _on_selection(self, event: Optional[Any] = None) -> None:
        self.x_widget.widget.clear()
        self.y_widget.widget.clear()
        self.color_widget.widget.clear()

        self.x_widget.widget._onChange()
        self.x_widget.component_widget._onChange()
        self.y_widget.widget._onChange()
        self.y_widget.component_widget._onChange()
        self.color_widget.widget._onChange()
        self.color_widget.component_widget._onChange()

    def _select_layer(self) -> None:
        """Napari layers."""
        layer = self._viewer.layers.selection._current
        if not isinstance(layer.metadata.get("adata", None), AnnData):
            raise NotImplementedError(":class:`anndata.AnnData` not found in any `layer.metadata`.")

        self.model.layer = layer
        # if layer is not None and "adata" in layer.metadata:
        self.model.adata = layer.metadata["adata"]

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
        """Names of :class:`napari.layers.Layer`."""
        return frozenset(layer.name for layer in self.viewer.layers)


class QtAdataViewWidget(QWidget):
    """Adata viewer widget."""

    def __init__(self, viewer: Viewer):
        super().__init__()

        self._viewer = viewer
        self._model = ImageModel()

        self._select_layer()
        self._viewer.layers.selection.events.changed.connect(self._select_layer)
        self._viewer.layers.selection.events.changed.connect(self._on_layer_update)

        self.setLayout(QVBoxLayout())

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
        self.obsm_index_widget = ComponentWidget(self.model, attr="obsm", max_visible=6)
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
        colorbar = CBarWidget(model=self.model)
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

    def _select_layer(self) -> None:
        """Napari layers."""
        layer = self._viewer.layers.selection._current
        if not isinstance(layer.metadata.get("adata", None), AnnData):
            raise NotImplementedError(":class:`anndata.AnnData` not found in any `layer.metadata`.")

        self.model.layer = layer
        # if layer is not None and "adata" in layer.metadata:
        self.model.adata = layer.metadata["adata"]

        if self.model.adata.shape == (0, 0):
            return

        self.model.coordinates = np.insert(self.model.adata.obsm[Key.obsm.spatial][:, ::-1][:, :2], 0, values=0, axis=1)
        if "points" in layer.metadata:
            self.model.points_coordinates = layer.metadata["points"].X
            self.model.points_var = layer.metadata["points"].obs["gene"]
            self.model.point_diameter = np.array([0.0] + [layer.metadata["point_diameter"]] * 2) * self.model.scale
        self.model.spot_diameter = np.array([0.0, 10.0, 10.0])
        self.model.labels_key = layer.metadata["labels_key"] if isinstance(layer, Labels) else None
        if "colormap" in layer.metadata:
            self.model.cmap = layer.metadata["colormap"]

    def _get_adata_layer(self) -> Sequence[Optional[str]]:
        adata_layers = list(self.model.adata.layers.keys())
        if len(adata_layers):
            return adata_layers
        return [None]

    def export(self, _: napari.viewer.Viewer) -> None:
        """Export shapes into :class:`anndata.AnnData` object."""
        for layer in self.viewer.layers:
            if not isinstance(layer, napari.layers.Shapes) or layer not in self.viewer.layers.selection:
                continue
            if not len(layer.data):
                logger.warn(f"Shape layer `{layer.name}` has no visible shapes.")
                continue

            key = f"{layer.name}_{self.model.layer.name}"  # type:ignore[union-attr]

            logger.info(f"Adding `adata.obs[{key!r}]`\n       `adata.uns[{key!r}]['mesh']`.")
            self._save_shapes(layer, key=key)
            self._update_obs_items(key)

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
        """Names of :class:`napari.layers.Layer`."""
        return frozenset(layer.name for layer in self.viewer.layers)
