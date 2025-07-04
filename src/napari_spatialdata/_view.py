from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import napari
import numpy as np
import pandas as pd
import xarray
from anndata import AnnData
from loguru import logger
from matplotlib.colors import to_rgba_array
from napari._qt.qt_resources import get_stylesheet
from napari._qt.utils import QImg2array
from napari.layers import Image, Labels, Layer, Points, Shapes
from napari.layers._multiscale_data import MultiScaleData
from napari.utils.events import Event
from napari.utils.notifications import show_info
from napari.viewer import Viewer
from pandas.api.types import CategoricalDtype
from qtpy import QtWidgets
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QGridLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from spatialdata import SpatialData, get_element_annotators, join_spatialelement_table
from spatialdata.models import TableModel

from napari_spatialdata._annotationwidgets import MainWindow
from napari_spatialdata._model import DataModel
from napari_spatialdata._scatterwidgets import AxisWidgets, PlotWidget
from napari_spatialdata._widgets import (
    AListWidget,
    AnnDataSaveDialog,
    CBarWidget,
    ComponentWidget,
    RangeSliderWidget,
    SaveDialog,
    ScatterAnnotationDialog,
)

__all__ = ["QtAdataViewWidget", "QtAdataScatterWidget"]

from napari_spatialdata.utils._utils import _get_init_table_list, block_signals


class QtAdataScatterWidget(QWidget):
    """Adata viewer widget."""

    def __init__(
        self, napari_viewer: Viewer | None = None, model: DataModel | None = None, adata: AnnData | None = None
    ):
        super().__init__()

        if napari_viewer is not None:
            self._viewer = napari_viewer

            self._model = (
                model
                if model is not None
                else self._viewer.window._dock_widgets["SpatialData"].widget().viewer_model._model
            )

            self._select_layer()
            self._viewer.layers.selection.events.changed.connect(self._select_layer)
            self._viewer.layers.selection.events.changed.connect(self._on_selection)

        elif adata is not None:

            self._viewer = None

            self._model = DataModel()
            self._model.adata = adata

            # fake an instance key as we don't have a napari layer
            col = self._model.adata.obs.columns[0]
            temp = {"region": "na", "region_key": "na", "instance_key": col}
            self._model.adata.uns["spatialdata_attrs"] = temp

            self.setStyleSheet(get_stylesheet("dark"))

        else:
            raise ValueError("Either napari viewer or adata object must be provided.")

        # create the layout
        self.setLayout(QGridLayout())

        # Create the splitter
        splitter = QSplitter(Qt.Vertical)

        # Plot widget
        self.plot_widget = PlotWidget(self.viewer, self.model)
        splitter.addWidget(self.plot_widget)

        # Control widget
        control_widget = QWidget()
        control_layout = QGridLayout()
        control_widget.setLayout(control_layout)

        if self._viewer is not None:

            table_label = QLabel("Tables annotating layer:")
            self.table_name_widget = QComboBox()
            if (table_names := self.model.table_names) is not None:
                self.table_name_widget.addItems(table_names)

            self.table_name_widget.currentTextChanged.connect(self._update_adata)
            control_layout.addWidget(table_label, 0, 0, Qt.AlignLeft)
            control_layout.addWidget(self.table_name_widget, 0, 1, 1, 2)

        self.x_widget = AxisWidgets(self.model, "X-axis")
        control_layout.addWidget(self.x_widget, 1, 0, 1, 1)

        self.y_widget = AxisWidgets(self.model, "Y-axis")
        control_layout.addWidget(self.y_widget, 1, 1, 1, 1)

        self.color_widget = AxisWidgets(self.model, "Color", True)
        control_layout.addWidget(self.color_widget, 1, 2, 1, 1)

        self.plot_button_widget = QPushButton("Plot")
        self.plot_button_widget.clicked.connect(
            lambda: self.plot_widget._onClick(
                self.x_widget.widget.data,
                self.y_widget.widget.data,
                self.color_widget.widget.data,
                self.x_widget.getFormattedLabel(),
                self.y_widget.getFormattedLabel(),
                self.color_widget.getFormattedLabel(),
            )
        )

        self.annotate_button_widget = QPushButton("Annotate")
        self.annotate_button_widget.clicked.connect(self.export)

        self.save_button_widget = QPushButton("Save")
        self.save_button_widget.clicked.connect(self.save_sdata)

        control_layout.addWidget(self.plot_button_widget, 2, 0, 1, 1)
        control_layout.addWidget(self.annotate_button_widget, 2, 1, 1, 1)
        control_layout.addWidget(self.save_button_widget, 2, 2, 1, 1)

        self.status_label = QLabel("Status: ")
        control_layout.addWidget(self.status_label, 3, 0, 1, 3)

        if self._viewer is not None:
            if self._viewer.layers.selection.active is not None:
                if "sdata" in self._viewer.layers.selection.active.metadata:
                    if self._viewer.layers.selection.active.metadata["sdata"].is_backed():
                        self.change_status("Sdata is backed - annotations can be saved.")
                    else:
                        self.change_status("Sdata present but not backed - annotations can be created but not saved.")
                else:
                    self.change_status("No sdata associated with the layer.")
            else:
                self.change_status("No layer selected.")
        else:
            self.change_status("No napari viewer detected. You can save annotations to AnnData directly.")

        splitter.addWidget(control_widget)

        # Add the splitter to the main layout
        self.layout().addWidget(splitter, 1, 0, 1, 3)

        self.model.events.adata.connect(self._on_selection)

    def change_status(self, new_status: str) -> None:
        """Change the status label text."""
        self.status_label.setText(f"Status: {new_status}")

    def open_annotation_dialog(self) -> str:
        dialog = ScatterAnnotationDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            annotation_name = dialog.get_annotation_name()

        return str(annotation_name)

    def export(self) -> None:
        """Export selected points as adata column."""
        if len(self.plot_widget.roi_list) > 0:

            # display a message window to get the column name
            self.annotation_name = self.open_annotation_dialog()

            if self.annotation_name != "":

                new_name = self.annotation_name not in self.model.adata.obs.columns

                if not new_name:
                    reply = QtWidgets.QMessageBox.question(
                        self,
                        "Overwrite Annotation",
                        f"The annotation'{self.annotation_name}' already exists. Do you want to overwrite it?",
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                        QtWidgets.QMessageBox.No,
                    )
                    if reply == QtWidgets.QMessageBox.No:
                        return

                logger.debug(f"Annotating selected points as {self.annotation_name}")

                # get selected points
                self.selected_vector = self.plot_widget.get_selection()

                # modify anndata table
                self.model.adata.obs[self.annotation_name] = self.selected_vector

                # modify sdata of the viewer
                if self._viewer is not None:
                    selected_layer = self._viewer.layers.selection.active
                    selected_table = self.table_name_widget.currentText()

                    # change adata of the layer
                    selected_layer.metadata["adata"] = self.model.adata

                    # change sdata of the layer
                    sel_obs = selected_layer.metadata["sdata"][selected_table].obs

                    if not new_name:
                        logger.debug("Dropping a column")
                        # clear the old annotation column
                        sel_obs = sel_obs.drop(self.annotation_name, axis=1)

                    columns_to_add = [col for col in self.model.adata.obs.columns if col not in sel_obs.columns]
                    merge_on = [
                        self.model.adata.uns["spatialdata_attrs"]["region_key"],
                        self.model.adata.uns["spatialdata_attrs"]["instance_key"],
                    ]

                    new_df = pd.merge(sel_obs, self.model.adata.obs[merge_on + columns_to_add], on=merge_on, how="left")
                    new_df[self.annotation_name] = new_df[self.annotation_name].astype("category")
                    selected_layer.metadata["sdata"][selected_table].obs = new_df

                    # check that the view widget is present
                    # and trigger its update
                    if "View (napari-spatialdata)" in dict(self._viewer.window._dock_widgets.items()):
                        view_widget = dict(self._viewer.window._dock_widgets.items())["View (napari-spatialdata)"]
                        qtAdataViewWidget = [x for x in view_widget.children() if isinstance(x, QtAdataViewWidget)][0]
                        qtAdataViewWidget._on_layer_update()

                # new annotation - update options
                if new_name:
                    # add the new option to the obs widget
                    # without changing the current selection
                    widgets = {
                        "color_widget": self.color_widget.widget,
                        "x_widget": self.x_widget.widget,
                        "y_widget": self.y_widget.widget,
                    }

                    for widget in widgets.values():
                        if widget.getAttribute() == "obs":
                            widget.addItems(self.annotation_name)

                    self.change_status("Annotation added.")

                # old annotation - change data and replot
                else:
                    # change widget data if the annotation already exists and is selected
                    if (self.color_widget.widget.getAttribute() == "obs") and (
                        self.color_widget.widget.chosen == self.annotation_name
                    ):
                        if self.color_widget.widget.data is not None:
                            self.color_widget.widget.data["vec"] = self.selected_vector

                        color_label = self.color_widget.getFormattedLabel()
                        if color_label is not None:
                            color_label = color_label + "__change__"

                        # automatically replot - may revisit later for performance
                        self.plot_widget._onClick(
                            self.x_widget.widget.data,
                            self.y_widget.widget.data,
                            self.color_widget.widget.data,
                            self.x_widget.getFormattedLabel(),
                            self.y_widget.getFormattedLabel(),
                            color_label,
                        )

                    self.change_status("Annotation updated.")

            # display status message that no column name was provided
            else:

                self.change_status("No column name provided.")

        # display status message - no rois
        else:

            self.change_status("No rois selected.")

    def save_sdata(self) -> None:
        """Save sdata or AnnData."""
        # data taken from the viewer
        if self._viewer is not None:

            selected_layer = self._viewer.layers.selection.active
            selected_table = self.table_name_widget.currentText()

            # trigger saving of the table
            # TODO: have to find another way to pass this as this is deprecated from 0.5.0 onwards
            # it's used in the save_sdata method
            self._viewer_model = self._viewer.window._dock_widgets["SpatialData"].widget().viewer_model
            self._viewer_model._write_element_to_disk(
                selected_layer.metadata["sdata"],
                selected_table,
                selected_layer.metadata["sdata"][selected_table],
                overwrite=True,
            )
        # data provided directly as AnnData
        else:
            d = AnnDataSaveDialog()
            file_path = d.show_dialog()

            if file_path:
                if file_path.endswith(".h5ad") or file_path.endswith(".zarr") or file_path.endswith(".csv"):
                    if file_path.endswith(".h5ad"):
                        self.model.adata.write(file_path)
                    if file_path.endswith(".zarr"):
                        self.model.adata.write_zarr(file_path)
                    if file_path.endswith(".csv"):
                        self.model.adata.write_csvs(file_path)
                    # display status message that AnnData was saved
                    self.change_status("AnnData saved!")

                else:
                    logger.info("File format not supported.")
                    # display status message that no valid file format was provided
                    self.change_status("Data not saved. Only h5ad, zarr and csv are supported.")

    def _update_adata(self) -> None:
        if (table_name := self.table_name_widget.currentText()) == "":
            return
        self.model.active_table_name = table_name
        layer = self._viewer.layers.selection.active

        if sdata := layer.metadata.get("sdata"):
            element_name = layer.metadata.get("name")
            _, table = join_spatialelement_table(
                sdata=sdata, spatial_element_names=element_name, table_name=table_name, how="left"
            )
            layer.metadata["adata"] = table

        if layer is not None:
            if "adata" in layer.metadata:
                with self.model.events.adata.blocker():
                    self.model.adata = layer.metadata["adata"]
            else:
                with self.model.events.adata.blocker():
                    self.model.adata = None

        if self.model.adata.shape == (0, 0):
            return

        self.model.system_name = layer.metadata.get("name", None)

        self.x_widget.widget._onChange()
        self.x_widget.component_widget._onChange()
        self.y_widget.widget._onChange()
        self.y_widget.component_widget._onChange()
        self.color_widget.widget._onChange()
        self.color_widget.component_widget._onChange()

    def _on_selection(self, event: Any) -> None:
        self.x_widget.widget.clear()
        self.y_widget.widget.clear()
        self.color_widget.widget.clear()

        self.table_name_widget.clear()
        self.table_name_widget.clear()
        if event.source == self.model or event.source.active:
            table_list = _get_init_table_list(self.viewer.layers.selection.active)
            if table_list:
                self.model.table_names = table_list
                self.table_name_widget.addItems(table_list)
                widget_index = self.table_name_widget.findText(table_list[0])
                self.table_name_widget.setCurrentIndex(widget_index)
        self.x_widget.widget._onChange()
        self.x_widget.component_widget._onChange()
        self.y_widget.widget._onChange()
        self.y_widget.component_widget._onChange()
        self.color_widget.widget._onChange()
        self.color_widget.component_widget._onChange()

    def _select_layer(self) -> None:
        """Napari layers."""
        layer = self._viewer.layers.selection.active
        self.model.layer = layer
        if not hasattr(layer, "metadata") or not isinstance(layer.metadata.get("adata"), AnnData):
            if hasattr(self, "x_widget"):
                self.table_name_widget.clear()
                self.x_widget.clear()
                self.y_widget.clear()
                self.color_widget.clear()
            return

        if layer is not None:
            self.model.adata = layer.metadata.get("adata", None)

    def screenshot(self) -> Any:
        return QImg2array(self.grab().toImage())

    @property
    def viewer(self) -> napari.Viewer:
        """:mod:`napari` viewer."""
        return self._viewer

    @property
    def model(self) -> DataModel:
        """:mod:`napari` viewer."""
        return self._model


class QtAdataViewWidget(QWidget):
    """Adata viewer widget."""

    def __init__(self, napari_viewer: Viewer, model: DataModel | None = None) -> None:
        super().__init__()

        self._viewer = napari_viewer
        self._model = model if model else napari_viewer.window.dock_widgets["SpatialData"].viewer_model.model

        self._select_layer()
        self._viewer.layers.selection.events.changed.connect(self._select_layer)

        self.setLayout(QVBoxLayout())

        # Names of tables annotating respective layer.
        table_label = QLabel("Tables annotating layer:")
        self.table_name_widget = QComboBox()
        if (table_names := self.model.table_names) is not None:
            self.table_name_widget.addItems(table_names)

        self.table_name_widget.currentTextChanged.connect(self._update_adata)
        self.layout().addWidget(table_label)
        self.layout().addWidget(self.table_name_widget)

        # obs
        obs_label = QLabel("Observations:")
        obs_label.setToolTip("Keys in `adata.obs` containing cell observations.")
        self.obs_widget = AListWidget(self.viewer, self.model, attr="obs")
        self.layout().addWidget(obs_label)
        self.layout().addWidget(self.obs_widget)

        # Vars
        var_label = QLabel("Vars:")
        var_label.setToolTip("Names from `adata.var_names` or `adata.raw.var_names`.")
        self.var_widget = AListWidget(self.viewer, self.model, attr="var")
        self.var_widget.setAdataLayer("X")

        self.viewer.dims.events.current_step.connect(self._channel_changed)

        # layers
        adata_layer_label = QLabel("Layers:")
        adata_layer_label.setToolTip("Keys in `adata.layers` used when visualizing gene expression.")
        self.adata_layer_widget = QComboBox()
        if self.model.adata is not None:
            self.adata_layer_widget.addItem("X", None)
            self.adata_layer_widget.addItems(self._get_adata_layer())

        self.adata_layer_widget.currentTextChanged.connect(self.var_widget.setAdataLayer)

        self.layout().addWidget(adata_layer_label)
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

        # Dataframe columns columns
        dataframe_columns_label = QLabel("Dataframe columns:")
        dataframe_columns_label.setToolTip("Columns in points/shapes element excluding dimension columns.")
        self.dataframe_columns_widget = AListWidget(self.viewer, self.model, attr="columns_df", multiselect=False)
        self.layout().addWidget(dataframe_columns_label)
        self.layout().addWidget(self.dataframe_columns_widget)

        # color by
        self.color_by = QLabel("Colored by:")
        self.layout().addWidget(self.color_by)

        # scalebar
        colorbar = CBarWidget(model=self.model)
        self.slider = RangeSliderWidget(self.viewer, self.model, colorbar=colorbar)
        self._viewer.window.add_dock_widget(self.slider, area="left", name="slider")
        self._viewer.window.add_dock_widget(colorbar, area="left", name="colorbar")
        self.viewer.layers.selection.events.active.connect(self.slider._onLayerChange)

        if (layer := self.viewer.layers.selection.active) is not None and layer.metadata.get("adata") is not None:
            self._on_layer_update()

        self.model.events.adata.connect(self._on_layer_update)
        self.model.events.color_by.connect(self._change_color_by)

    def _channel_changed(self, event: Event) -> None:
        layer = self.model.layer
        is_image = isinstance(layer, Image)

        has_sdata = layer is not None and layer.metadata.get("sdata") is not None
        has_adata = layer is not None and layer.metadata.get("adata") is not None

        # has_adata is added so we see the channels in the view widget under vars
        if layer is None or not is_image or not has_sdata or not has_adata or layer.rgb:
            return

        current_point = list(event.value)
        displayed = self._viewer.dims.displayed
        if layer.multiscale:
            for i, (lo_size, hi_size, cord) in enumerate(
                zip(layer.data[-1].shape, layer.data[0].shape, current_point, strict=False)
            ):
                if i in displayed:
                    current_point[i] = slice(None)
                else:
                    current_point[i] = int(cord * lo_size / hi_size)
        else:
            for i in range(len(current_point)):
                if i in displayed:
                    current_point[i] = slice(None)
        # TODO remove once contrast limits in napari are fixed
        if isinstance(layer.data, MultiScaleData):
            # just compute lowest resolution
            image = layer.data[-1][tuple(current_point)].compute()
            min_value = image.min().data
            max_value = image.max().data
        else:
            image = layer.data[tuple(current_point)].compute()
            min_value = image.min()
            max_value = image.max()

        if isinstance(min_value, xarray.DataArray):
            min_value = min_value.item()
        if isinstance(max_value, xarray.DataArray):
            max_value = max_value.item()

        if min_value == max_value:
            min_value = np.iinfo(image.data.dtype).min
            max_value = np.iinfo(image.data.dtype).max
        layer.contrast_limits = [min_value, max_value]
        try:
            channel_num = next(x for x in current_point if not isinstance(x, slice))
        except StopIteration:
            return

        item = self.var_widget.item(channel_num)
        index = self.var_widget.indexFromItem(item)
        self.var_widget.setCurrentIndex(index)

    def _on_layer_update(self, event: Any | None = None) -> None:
        """When the model updates the selected layer, update the relevant widgets."""
        logger.debug("Updating layer.")

        self.table_name_widget.clear()

        table_list = _get_init_table_list(self.viewer.layers.selection.active)
        if table_list:
            self.model.table_names = table_list
            self.table_name_widget.addItems(table_list)
            widget_index = self.table_name_widget.findText(table_list[0])
            self.table_name_widget.setCurrentIndex(widget_index)
        self.adata_layer_widget.clear()
        self.adata_layer_widget.addItem("X", None)
        self.adata_layer_widget.addItems(self._get_adata_layer())
        self.dataframe_columns_widget.clear()
        if self.model.layer is not None and (cols_df := self.model.layer.metadata.get("_columns_df")) is not None:
            self.dataframe_columns_widget.addItems(map(str, cols_df.columns))
        self.obs_widget._onChange()
        self.var_widget._onChange()
        self.obsm_widget._onChange()

    def _select_layer(self) -> None:
        """Napari layers."""
        layer = self._viewer.layers.selection.active
        self.model.layer = layer
        if not hasattr(layer, "metadata") or not isinstance(layer.metadata.get("adata", None), AnnData):
            if hasattr(self, "obs_widget"):
                self.table_name_widget.clear()
                self.adata_layer_widget.clear()
                self.dataframe_columns_widget.clear()
                self.obs_widget.clear()
                self.var_widget.clear()
                self.obsm_widget.clear()
                self.color_by.clear()
                if isinstance(layer, Points | Shapes) and (cols_df := layer.metadata.get("_columns_df")) is not None:
                    self.dataframe_columns_widget.addItems(map(str, cols_df.columns))
                    self.model.system_name = layer.metadata.get("name", None)
            self.model.adata = None
            return

        if layer is not None:
            self.model.adata = layer.metadata.get("adata", None)

        if self.model.adata.shape == (0, 0):
            return

        self.model._region_key = layer.metadata["region_key"] if isinstance(layer, Labels) else None
        self.model._instance_key = layer.metadata["instance_key"] if isinstance(layer, Labels) else None
        self.model.system_name = layer.metadata.get("name", None)

        if hasattr(
            self, "obs_widget"
        ):  # to check if the widget has been already initialized, layer update should only be called on layer change
            self._on_layer_update()
        else:
            return

    def _update_adata(self) -> None:
        if (table_name := self.table_name_widget.currentText()) == "":
            return
        self.model.active_table_name = table_name

        layer = self._viewer.layers.selection.active

        if sdata := layer.metadata.get("sdata"):
            element_name = layer.metadata.get("name")
            how = "left" if isinstance(layer, Labels) else "inner"
            _, table = join_spatialelement_table(
                sdata=sdata, spatial_element_names=element_name, table_name=table_name, how=how
            )
            layer.metadata["adata"] = table

        if layer is not None:
            if "adata" in layer.metadata:
                with self.model.events.adata.blocker():
                    self.model.adata = layer.metadata["adata"]
            else:
                with self.model.events.adata.blocker():
                    self.model.adata = None

        if self.model.adata is None or self.model.adata.shape == (0, 0):
            return

        self.model.system_name = layer.metadata.get("name", None)

        if hasattr(
            self, "obs_widget"
        ):  # to check if the widget has been already initialized, layer update should only be called on layer change
            self.adata_layer_widget.clear()
            self.adata_layer_widget.addItem("X", None)
            self.adata_layer_widget.addItems(self._get_adata_layer())
            self.obs_widget._onChange()
            self.var_widget._onChange()
            self.obsm_widget._onChange()
        else:
            return

    def _get_adata_layer(self) -> Sequence[str | None]:
        if self.model.adata is None:
            return [None]
        adata_layers = list(self.model.adata.layers.keys())
        if len(adata_layers) == 0:
            return [None]
        return adata_layers

    def _change_color_by(self) -> None:
        self.color_by.setText(f"Color by: {self.model.color_by}")

    @property
    def viewer(self) -> napari.Viewer:
        """:mod:`napari` viewer."""
        return self._viewer

    @property
    def model(self) -> DataModel:
        """:mod:`napari` viewer."""
        return self._model

    @property
    def layernames(self) -> frozenset[str]:
        """Names of :class:`napari.layers.Layer`."""
        return frozenset(layer.name for layer in self.viewer.layers)


class QtAdataAnnotationWidget(QWidget):
    """Adata annotation widget."""

    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self._viewer = napari_viewer
        # TODO: have to find another way to pass this as this is deprecated from 0.5.0 onwards
        self._viewer_model = self._viewer.window._dock_widgets["SpatialData"].widget().viewer_model

        self.setLayout(QGridLayout())
        self._current_color = "#FFFFFF"
        self._current_class = "undefined"
        self._current_annotator = ""
        self._current_description = ""
        self._current_region = None
        self._current_region_key = "region"
        self._current_instance_key = "instance_id"
        self._line_edit_to_class_name: dict[QLineEdit, str] = {}

        self.annotation_widget = MainWindow()
        self.layout().addWidget(self.annotation_widget)
        self.annotation_widget.tree_view.button_group.buttonClicked.connect(self._on_class_radio_click)
        self.annotation_widget.annotators.currentTextChanged.connect(self._set_current_annotator)
        self.annotation_widget.table_name_widget.currentTextChanged.connect(self._import_table_information)
        self.annotation_widget.save_button.clicked.connect(self._open_save_dialog)
        self.annotation_widget.tree_view.header().sectionClicked.connect(self._header_clicked)
        self.annotation_widget.tree_view.model.headerDataChanged.connect(self._change_class_column_name)
        self.annotation_widget.set_annotation.clicked.connect(self._set_class_description)
        self.annotation_widget.description_box.textChanged.connect(self._set_current_description)
        self.annotation_widget.tree_view.color_button_added.connect(self._connect_button_to_change_color)
        self.annotation_widget.tree_view.class_name_text_added.connect(self._connect_line_edit_change_class)
        self._current_class_column = self.annotation_widget.tree_view.model.horizontalHeaderItem(2).text()
        self._set_editable_save_button()
        self._set_clickable_buttons()

        self.viewer.layers.events.inserted.connect(self._on_inserted)
        self.viewer.layers.events.inserted.connect(self._set_editable_save_button)
        self.viewer.layers.events.inserted.connect(self._set_clickable_buttons)
        self.viewer.layers.selection.events.changed.connect(self._on_layer_selection_changed)
        self.viewer.layers.selection.events.changed.connect(self._set_editable_save_button)
        self.viewer.layers.selection.events.changed.connect(self._set_clickable_buttons)
        self.annotation_widget.link_button.clicked.connect(self._link_layer)
        self._on_layer_selection_changed()

        if isinstance(layer := self.viewer.layers.selection.active, Shapes):
            layer.events.name.connect(self._change_region_on_name_change)
            self._connect_layer_events(layer)

    def _connect_button_to_change_color(self, button: QPushButton) -> None:
        button.color_changed.connect(self._on_color_of_button_change)

    def _connect_line_edit_change_class(self, line_edit: QLineEdit) -> None:
        self._line_edit_to_class_name[line_edit] = line_edit.text()
        # TODO decide on out of focus override or not
        line_edit.returnPressed.connect(lambda: self._on_class_name_of_line_edit_change(line_edit))

    def _on_class_name_of_line_edit_change(self, line_edit: QLineEdit) -> None:
        layer = self.viewer.layers.selection.active
        class_name = line_edit.text()
        radio_button = self.annotation_widget.tree_view.button_group.checkedButton()

        class_name_ind = self.annotation_widget.tree_view.button_to_class_index[radio_button]
        previous_text = self._line_edit_to_class_name[line_edit]

        indices = layer.features.index[layer.features["class"] == previous_text].tolist()
        layer.features["class"] = layer.features["class"].cat.add_categories(class_name)
        layer.features.loc[indices, "class"] = class_name
        self._line_edit_to_class_name[line_edit] = class_name

        if line_edit == self.annotation_widget.tree_view.indexWidget(class_name_ind):
            self._current_class = class_name

    def _on_color_of_button_change(self, color: str, color_button: QPushButton) -> None:
        layer = self.viewer.layers.selection.active
        radio_button = self.annotation_widget.tree_view.button_group.checkedButton()
        # We have five columns with at position 1 the color button
        color_ind = self.annotation_widget.tree_view.button_to_color_index[radio_button]

        class_name = self.annotation_widget.tree_view.color_button_to_class_line_edit[color_button].text()
        indices = layer.features.index[layer.features["class"] == class_name].tolist()
        layer.features["class_color"] = layer.features["class_color"].cat.add_categories(color)
        layer.features.loc[indices, "class_color"] = color
        rgba_color = to_rgba_array(color)

        layer.selected_data = set(indices)
        layer.current_face_color = rgba_color
        layer.current_edge_color = rgba_color
        layer.selected_data = set()
        layer.current_face_color = self._current_color
        layer.current_edge_color = self._current_color

        if color_button == self.annotation_widget.tree_view.indexWidget(color_ind):
            self._viewer.layers.selection.active.current_face_color = color
            self._viewer.layers.selection.active.current_edge_color = color
            self._current_color = color

    def _set_current_description(self) -> None:
        self._current_description = self.annotation_widget.description_box.toPlainText()

    def _on_inserted(self, event: Event) -> None:
        """
        Update table name dropdown and set up feature df and face color update.

        Upon inserting a layer, first  it is checked whether it is a shapes layer that has sdata
        in the metadata. If that is the case it is checked whether any of the tables contains a color column and if
        so the table name widget is updated. Next, the layer features are set to an empty dataframe and the face
        color of the layer is set to the current color in the annotation widget.

        Parameters
        ----------
        event
            The napari event for a layer being inserted in the viewer layerlist.
        """
        layer = event.value
        self._connect_layer_events(layer)
        self._set_editable_save_button()

    def _connect_layer_events(self, layer: Shapes) -> None:
        if layer and isinstance(layer, Shapes):
            layer.events.data.connect(self._update_annotations)
            layer.events.name.connect(self._change_region_on_name_change)
            layer.mouse_move_callbacks.append(self._on_mouse_move)
            self._current_region = layer.name
            layer.current_face_color = self._current_color

    def _on_mouse_move(self, layer: Layer, event: Event) -> None:
        if layer == self.viewer.layers.selection.active:
            if (shape_index := layer.get_value(event.position)[0]) is not None:
                description = layer.features.loc[shape_index, "description"]
                annotator = layer.features.loc[shape_index, "annotator"]

                # Due to bug in napari with not respecting layer.features and different orders of events, description
                # and annotator in layer.features can be nan. Code here ensures that these will be str.
                description = description if isinstance(description, str) else self._current_description
                annotator = annotator if isinstance(annotator, str) else self._current_annotator

                with block_signals(self.annotation_widget.description_box):
                    self.annotation_widget.description_box.setText(description)
                with block_signals(self.annotation_widget.annotators):
                    self.annotation_widget.annotators.setCurrentText(annotator)
            else:
                self.annotation_widget.description_box.setText(self._current_description)
                self.annotation_widget.annotators.setCurrentText(self._current_annotator)

    def _on_layer_selection_changed(self) -> None:
        layer = self._viewer.layers.selection.active
        if isinstance(layer, Shapes):
            if sdata := layer.metadata.get("sdata"):
                if layer.name in sdata.shapes:
                    self._update_table_name_widget(sdata, layer.metadata["name"])
            else:
                self.annotation_widget.table_name_widget.clear()

            if self.annotation_widget.table_name_widget.currentText() == "" and len(layer.features.columns) == 0:
                self.annotation_widget.tree_view.reset_class_column_header()
                self.annotation_widget.tree_view.reset_to_default_tree_view()
                n_obs = len(layer.data)
                df = pd.DataFrame(
                    {
                        "class": pd.Series(pd.Categorical([self._current_class] * n_obs), dtype="category"),
                        "class_color": pd.Series([self._current_color] * n_obs, dtype="category"),
                        "description": pd.Series([""] * n_obs, dtype="str"),
                        "annotator": pd.Series([""] * n_obs, dtype="category"),
                        "region": pd.Series([layer.name] * n_obs, dtype="category"),
                    }
                )
                layer.features = df
                layer.metadata["annotation_region_key"] = self._current_region_key
                layer.metadata["annotation_instance_key"] = self._current_instance_key
                layer.feature_defaults = self._create_feature_default(layer)

            elif (
                len(color_cols := [col for col in layer.features.columns if "color" in col]) != 0
                and len(layer.features) != 0
            ):
                color_column_name = color_cols[0]
                class_column_name = color_column_name.split("_")[0]
                color_dict = layer.features.copy().set_index(class_column_name)[color_column_name].to_dict()

                for class_name, color in color_dict.items():
                    if class_name != "undefined":
                        self.annotation_widget.tree_view.addGroup(color=color, name=class_name)
                self.annotation_widget.annotators.addItems(layer.features["annotator"].cat.categories.to_list())
            self._current_region = layer.name

        else:
            self.annotation_widget.tree_view.reset_class_column_header()
            self.annotation_widget.tree_view.reset_to_default_tree_view()
            self.annotation_widget.table_name_widget.clear()
            self.annotation_widget.annotators.clear()
            self._current_region = None

        self._set_editable_save_button()

    @property
    def viewer(self) -> napari.Viewer:
        """:mod:`napari` viewer."""
        return self._viewer

    def _update_annotations(self, event: Event) -> None:
        """
        Update annotations in layer.features.

        When adding an element on the shapes layer, a new row with Nans is automatically added to layer.features by
        napari itself. This row needs to be replaced with the values we want. This function checks whether we added
        a single shape and then calls the respective function to update the last row of layer.features.

        event
            The napari event for changes being made on the layer, namely "adding", "added", "changing", "changed,
            "removing" or "removed". Here we only deal with "added" currently, but this could be changed later.
        """
        layer = event.source
        columns_to_convert = ["class", "class_color", "annotator", self._current_region_key]
        if not isinstance(layer.features["class"], CategoricalDtype):
            layer.features[columns_to_convert] = layer.features[columns_to_convert].astype("category")
        if event.action == "added" and len(event.data_indices) == 1:
            self._update_layer_features(layer, event.action)
        elif event.action == "added" and len(event.data_indices) > 1:
            raise ValueError(f"Can only add one annotation at the time, got {len(event.data_indices)}")

    def _update_layer_features(self, layer: Layer, action: str) -> None:
        """Add categories to respective column if not present yet and update last row of layer.features."""
        self._add_categories(layer)
        if action == "added":
            row = [
                self._current_class,
                self._current_color,
                self._current_description,
                self._current_annotator,
                self._current_region,
            ]
            layer.features.loc[len(layer.features) - 1] = row

        self._set_editable_save_button()

    def _add_categories(self, layer: Layer) -> None:
        """
        Add new categories of class and annotator.

        In order to be able to add a row value to a categorical series, the value should be added to the categories of
        that series otherwise it would result in an error. This code checks whether the current class of the annotation
        widget is already present as category in the respective columns and if not adds it.

        layer
        The napari shapes layer.
        """
        if self._current_class not in layer.features["class"].cat.categories:
            layer.features["class"] = layer.features["class"].cat.add_categories(self._current_class)
        if self._current_annotator not in layer.features["annotator"].cat.categories:
            layer.features["annotator"] = layer.features["annotator"].cat.add_categories(self._current_annotator)
        if self._current_color not in layer.features["class_color"].cat.categories:
            layer.features["class_color"] = layer.features["class_color"].cat.add_categories(self._current_color)
        if layer.name not in layer.features["region"].cat.categories:
            layer.features["region"] = layer.features["region"].cat.add_categories(self._current_region)

    def _create_feature_default(self, layer: Layer) -> pd.DataFrame:
        return pd.DataFrame(
            {
                #                "instance_id": pd.Series([], dtype="int"),
                "class": pd.Series(pd.Categorical(["undefined"]), dtype="category"),
                "class_color": pd.Series(["#FFFFFF"], dtype="category"),
                "description": pd.Series([""], dtype="str"),
                "annotator": pd.Series([""], dtype="category"),
                self._current_region_key: pd.Series([layer.name], dtype="category"),
            }
        )

    def _on_class_radio_click(self) -> None:
        layer = self.viewer.layers.selection.active
        if isinstance(layer, Shapes):
            radio_button = self.annotation_widget.tree_view.button_group.checkedButton()
            # We have five columns with at position 1 the color button
            color_ind = self.annotation_widget.tree_view.button_to_color_index[radio_button]
            color_button = self.annotation_widget.tree_view.indexWidget(color_ind)

            class_ind = self.annotation_widget.tree_view.button_to_class_index[radio_button]
            class_widget = self.annotation_widget.tree_view.indexWidget(class_ind)
            self._current_class = class_widget.text()

            # .name() converts the QColor in hexadecimal
            palette = color_button.palette()
            color = palette.color(color_button.backgroundRole()).name()
            self._current_color = color

            if layer.mode != "select":
                self._viewer.layers.selection.active.current_face_color = color
                self._viewer.layers.selection.active.current_edge_color = color

    def _set_current_annotator(self) -> None:
        """Update current annotator when the text of the annotator dropdown has changed."""
        self._current_annotator = self.annotation_widget.annotators.currentText()

    def _update_table_name_widget(self, sdata: SpatialData, element_name: str, table_name: str = "") -> None:
        table_names = list(get_element_annotators(sdata, element_name))

        table_names_to_add = []
        for name in table_names:
            if any("color" in key for key in sdata[name].uns):
                table_names_to_add.append(name)

        self.annotation_widget.table_name_widget.clear()
        self.annotation_widget.table_name_widget.addItems(table_names_to_add)
        if table_name != "":
            self.annotation_widget.table_name_widget.setCurrentText(table_name)

    def _import_table_information(self) -> None:
        table_name = self.annotation_widget.table_name_widget.currentText()
        layer = self.viewer.layers.selection.active
        if layer and table_name != "":
            # self.annotation_widget.tree_view.reset_to_default_tree_view()

            sdata = layer.metadata["sdata"]
            table = sdata[table_name]
            color_column = [key for key in table.uns if "color" in key][0]
            color_dict = table.uns[color_column]

            self._current_region_key = table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
            self._current_instance_key = table.uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY]
            layer.metadata["annotation_region_key"] = self._current_region_key
            layer.metadata["annotation_instance_key"] = self._current_instance_key

            feature_df = table.obs.copy().drop(columns=[self._current_instance_key])
            class_column = color_column.rpartition("_")[0]
            feature_df[color_column] = feature_df[class_column].map(color_dict)
            feature_df[color_column] = feature_df[color_column].astype("category")
            color_array = to_rgba_array(feature_df[color_column])
            layer.face_color = color_array
            layer.edge_color = color_array

            self._current_class_column = class_column
            self.annotation_widget.tree_view.set_class_column_header(class_column)

            if "description" not in feature_df.columns:
                feature_df["description"] = pd.Series([""] * len(feature_df), dtype="str", index=feature_df.index)

            if "annotator" in feature_df.columns:
                annotators = list(feature_df["annotator"].cat.categories)
                self.annotation_widget.annotators.clear()
                self.annotation_widget.annotators.addItems(annotators)
                self.annotation_widget.annotators.setCurrentText(annotators[0])
            else:
                feature_df["annotator"] = pd.Series([""] * len(feature_df), dtype="category", index=feature_df.index)

            feature_df.rename(columns={class_column: "class", class_column + "_color": "class_color"}, inplace=True)
            feature_df = feature_df.reindex(
                columns=["class", "class_color", "description", "annotator", self._current_region_key]
            )
            layer.features = feature_df

            # for class_name, color in color_dict.items():
            #     self.annotation_widget.tree_view.addGroup(color=color, name=class_name)

    def _open_save_dialog(self) -> None:
        layer = self.viewer.layers.selection.active
        previous_shape_name = layer.metadata["name"]
        save_dialog = SaveDialog(layer, self.annotation_widget.table_name_widget.currentText())
        save_dialog.exec_()
        table_name = save_dialog.get_save_table_name()
        shape_name = save_dialog.get_save_shape_name()
        if shape_name and table_name:
            self._viewer_model.save_to_sdata(
                [layer],
                shape_name,
                table_name,
                table_columns=[self._current_class_column, self._current_class_column + "_color"],
                overwrite=True,
            )
            if (
                (previous_table := self.annotation_widget.table_name_widget.currentText()) != table_name
                and previous_table != ""
                and shape_name == previous_shape_name
            ):
                del layer.metadata["sdata"].tables[previous_table]
                layer.metadata["sdata"].delete_element_from_disk(previous_table)
                show_info(f"Table name has changed and table with name {previous_table} has been deleted.")
            if previous_table == table_name and shape_name != previous_shape_name and previous_table != "":
                del layer.metadata["sdata"].shapes[previous_shape_name]
                layer.metadata["sdata"].delete_element_from_disk(previous_shape_name)
                self._viewer_model.layer_saved.emit(layer.metadata["_current_cs"])
            self._update_table_name_widget(layer.metadata["sdata"], layer.metadata["name"], table_name)
        else:
            show_info("Saving canceled.")

    def _set_editable_save_button(self) -> None:
        layer = self.viewer.layers.selection.active
        if not isinstance(layer, Shapes) or layer.metadata.get("sdata") is None or len(layer.features) == 0:
            self.annotation_widget.save_button.setEnabled(False)
        else:
            self.annotation_widget.save_button.setEnabled(True)

    def _set_clickable_buttons(self) -> None:
        layer = self.viewer.layers.selection.active
        if layer is not None and layer.metadata.get("sdata") is not None and not isinstance(layer, Image):
            self.annotation_widget.link_button.setEnabled(False)
            self.annotation_widget.add_button.setEnabled(True)
        else:
            self.annotation_widget.link_button.setEnabled(True)
            self.annotation_widget.add_button.setEnabled(False)

    def _link_layer(self) -> None:
        self._viewer_model._inherit_metadata(self._viewer, show_tooltip=True)
        self._set_clickable_buttons()
        self._on_layer_selection_changed()

        layer = self.viewer.layers.selection.active
        layer.events.data.connect(self._update_annotations)

    def _header_clicked(self, index: int) -> None:
        if index == 2:
            layer = self.viewer.layers.selection.active
            line_edit_widget = self.annotation_widget.tree_view.model.horizontalHeaderItem(2)
            current_text = line_edit_widget.text()
            new_text, ok = QInputDialog.getText(
                self, "Edit column name", "Enter new column name:", QLineEdit.Normal, current_text
            )
            if ok and isinstance(layer, Shapes) and any("color" in col for col in layer.features):
                # Set the new text to the line edit
                line_edit_widget.setText(new_text)
            else:
                show_info("Cannot change the color name when no annotation shapes layer is selected / active.")

    def _change_class_column_name(self) -> None:
        self._current_class_column = self.annotation_widget.tree_view.model.horizontalHeaderItem(2).text()

    def _set_class_description(self) -> None:
        layer = self.viewer.layers.selection.active
        if len(elements := layer.selected_data) == 1:
            self._add_categories(layer)
            self._viewer.layers.selection.active.current_face_color = self._current_color
            self._viewer.layers.selection.active.current_edge_color = self._current_color
            description = self.annotation_widget.description_box.toPlainText()
            self.annotation_widget.description_box.clear()
            self._current_description = ""
            annotator = self.annotation_widget.annotators.currentText()

            row = layer.features.loc[list(elements)]
            row["class"] = self._current_class
            row["class_color"] = self._current_color
            row["description"] = description
            row["annotator"] = annotator
            layer.features.loc[list(elements)] = row
        elif len(elements) >= 1:
            show_info("Can only set the description and class of one element at the time")
        else:
            show_info("No element in the layer selected")

    def _change_region_on_name_change(self, event: Event) -> None:
        self._current_region = event.source.name
        if len(event.source.features) != 0:
            event.source.features[self._current_region_key] = pd.Series(
                [self._current_region] * len(event.source.data), dtype="category"
            )
