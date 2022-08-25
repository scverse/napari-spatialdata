from typing import Any, Optional, Sequence, FrozenSet

from loguru import logger
from anndata import AnnData
from magicgui import magicgui
from napari.layers import Layer, Labels
from napari.viewer import Viewer
from qtpy.QtWidgets import QLabel, QWidget, QComboBox, QPushButton, QVBoxLayout
from napari_matplotlib.scatter import ScatterWidget
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

from napari_matplotlib.scatter import FeaturesScatterWidget, ScatterWidget, ScatterBaseWidget

__all__ = ["QtAdataViewWidget", "QtAdataScatterWidget"]


def test_data():
    from skimage.measure import regionprops_table

    # make a test label image
    label_image = np.zeros((100, 100), dtype=np.uint16)

    label_image[10:20, 10:20] = 1
    label_image[50:70, 50:70] = 2

    feature_table_1 = regionprops_table(label_image, properties=("label", "area", "perimeter"))
    feature_table_1["index"] = feature_table_1["label"]

    # make the points data
    n_points = 100
    points_data = 100 * np.random.random((100, 2))
    points_features = {
        "feature_0": np.random.random((n_points,)),
        "feature_1": np.random.random((n_points,)),
        "feature_2": np.random.random((n_points,)),
    }
    return label_image, feature_table_1, points_data, points_features


class QtAdataScatterWidget(QWidget):
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

        # Dropdown menu to select between obs, obsm, var for X axis
        x_selection_label = QLabel("Select type for X axis:")
        x_selection_label.setToolTip("Select between obs, obsm and var.")
        self.x_selection_widget = QComboBox()
        self.x_selection_widget.addItem("obsm", None)
        self.x_selection_widget.addItem("obs", None)
        self.x_selection_widget.addItem("var", None)

        self.layout().addWidget(x_selection_label)
        self.layout().addWidget(self.x_selection_widget)

        # X-axis
        x_label = QLabel("Select x-axis:")
        x_label.setToolTip("Select layer to visualise in x-axis.")

        self.x_widget = AListWidget(self.viewer, self.model, attr="obsm")
        self.x_widget.setAttribute("obsm")
        self.layout().addWidget(x_label)
        self.layout().addWidget(self.x_widget)

        self.x_selection_widget.currentTextChanged.connect(self.x_widget.setAttribute)

        # Y selection
        y_selection_label = QLabel("Select type for Y axis:")
        y_selection_label.setToolTip("Select between obs, obsm and var.")
        self.y_selection_widget = QComboBox()
        self.y_selection_widget.addItem("obsm", None)
        self.y_selection_widget.addItem("obs", None)
        self.y_selection_widget.addItem("var", None)

        self.layout().addWidget(y_selection_label)
        self.layout().addWidget(self.y_selection_widget)

        # Y-axis
        y_label = QLabel("Select y-axis:")
        y_label.setToolTip("Select layer to visualise in y-axis.")

        self.y_widget = AListWidget(self.viewer, self.model, attr="obsm")
        self.y_widget.setAttribute("obsm")
        self.layout().addWidget(y_label)
        self.layout().addWidget(self.y_widget)

        self.y_selection_widget.currentTextChanged.connect(self.y_widget.setAttribute)

        # Color
        color_selection_label = QLabel("Select type for color:")
        color_selection_label.setToolTip("Select between obs and var.")
        self.color_selection_widget = QComboBox()
        self.color_selection_widget.addItem("obs", None)
        self.color_selection_widget.addItem("var", None)

        self.layout().addWidget(color_selection_label)
        self.layout().addWidget(self.color_selection_widget)

        color_label = QLabel("Select color:")
        color_label.setToolTip("Select color to visualise the scatterplot.")
        self.color_widget = AListWidget(self.viewer, self.model, attr="obsm")
        self.color_widget.setAttribute("obs")
        self.layout().addWidget(color_label)
        self.layout().addWidget(self.color_widget)

        self.color_selection_widget.currentTextChanged.connect(self.color_widget.setAttribute)

        ###

        label_image, feature_table_1, points_data, points_features = test_data()
        # self._viewer.add_labels(label_image, features=feature_table_1)
        # self._viewer.add_points(points_data, features=points_features)

        ###
        self._viewer.add_labels(label_image)
        self._viewer.add_points(points_data)

        self.graph_widget = ScatterWidget(self._viewer)

        self.plot_button_widget = QPushButton("Plot")

        self.layout().addWidget(color_label)
        self.layout().addWidget(self.color_widget)

        self.layout().addWidget(self.graph_widget)
        self.layout().addWidget(self.plot_button_widget)

        self.model.events.adata.connect(self._on_selection)

    def _on_selection(self, event: Optional[Any] = None) -> None:
        self.x_widget.clear()
        self.y_widget.clear()
        self.color_widget.clear()
        self.x_widget._onChange()
        self.y_widget._onChange()
        self.color_widget._onChange()

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
        logger.info("Updating layer.")

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
                logger.warn(f"Shape layer `{layer.name}` has no visible shapes.")
                continue

            key = f"{layer.name}_{self.model.layer.name}"

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
        """Names of :attr:`napari.Viewer.layers`."""
        return frozenset(layer.name for layer in self.viewer.layers)
