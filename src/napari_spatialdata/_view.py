from typing import Any, FrozenSet, Optional, Sequence

import napari
import pandas as pd
from anndata import AnnData
from loguru import logger
from napari._qt.qt_resources import get_stylesheet
from napari._qt.utils import QImg2array
from napari.layers import Labels, Points, Shapes
from napari.viewer import Viewer
from qtpy.QtCore import QSize, Qt
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from spatialdata import join_spatialelement_table

from napari_spatialdata._annotationwidgets import MainWindow
from napari_spatialdata._model import DataModel, LayerFeatureModel
from napari_spatialdata._scatterwidgets import AxisWidgets, MatplotlibWidget
from napari_spatialdata._widgets import (
    AListWidget,
    CBarWidget,
    ComponentWidget,
    RangeSliderWidget,
)

__all__ = ["QtAdataViewWidget", "QtAdataScatterWidget", "QtAdataAnnotationWidget"]

from napari_spatialdata.utils._utils import _get_init_table_list


class QtAdataScatterWidget(QWidget):
    """Adata viewer widget."""

    def __init__(self, input: Viewer):
        super().__init__()

        self._model = DataModel()

        self.setLayout(QGridLayout())

        if isinstance(input, Viewer):
            self._viewer = input
            self._select_layer()
            self._viewer.layers.selection.events.changed.connect(self._select_layer)
            self._viewer.layers.selection.events.changed.connect(self._on_selection)

        elif isinstance(input, AnnData):
            self._viewer = None
            self.model.adata = input
            self.setStyleSheet(get_stylesheet("dark"))
            self.quit_button_widget = QPushButton("Close")
            self.quit_button_widget.clicked.connect(self.close)
            self.quit_button_widget.setStyleSheet("background-color: red")
            self.quit_button_widget.setFixedSize(QSize(100, 25))
            self.layout().addWidget(self.quit_button_widget, 0, 2, 1, 1, Qt.AlignRight)

        # Matplotlib

        self.matplotlib_widget = MatplotlibWidget(self.viewer, self.model)
        self.layout().addWidget(self.matplotlib_widget, 1, 0, 1, 3)

        # Names of tables annotating respective layer.
        table_label = QLabel("Tables annotating layer:")
        self.table_name_widget = QComboBox()
        if (table_names := self.model.table_names) is not None:
            self.table_name_widget.addItems(table_names)

        self.table_name_widget.currentTextChanged.connect(self._update_adata)
        self.layout().addWidget(table_label, 2, 0, Qt.AlignLeft)
        self.layout().addWidget(self.table_name_widget)

        self.x_widget = AxisWidgets(self.model, "X-axis")
        self.layout().addWidget(self.x_widget, 3, 0, 6, 1)

        self.y_widget = AxisWidgets(self.model, "Y-axis")
        self.layout().addWidget(self.y_widget, 3, 1, 6, 1)

        self.color_widget = AxisWidgets(self.model, "Color", True)
        self.layout().addWidget(self.color_widget, 3, 2, 6, 1)

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

        self.layout().addWidget(self.plot_button_widget, 9, 0, 1, 2)
        self.layout().addWidget(self.export_button_widget, 9, 2, 1, 2)

        self.model.events.adata.connect(self._on_selection)

    def export(self) -> None:
        """Export shapes."""
        if (self.matplotlib_widget.selector) is None or (self.matplotlib_widget.selector.exported_data is None):
            raise ValueError("Data points haven't been selected from the matplotlib visualisation.")

        self.matplotlib_widget.selector.export(self.model.adata)

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

        if layer is not None and "adata" in layer.metadata:
            with self.model.events.adata.blocker():
                self.model.adata = layer.metadata["adata"]

        if self.model.adata.shape == (0, 0):
            return

        self.model.instance_key = layer.metadata["instance_key"] = (
            table.uns["spatialdata_attrs"]["instance_key"] if table is not None else None
        )
        self.model.region_key = layer.metadata["region_key"] = (
            table.uns["spatialdata_attrs"]["region_key"] if table is not None else None
        )
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

    def _get_init_table_list(self) -> Optional[Sequence[Optional[str]]]:
        layer = self.viewer.layers.selection.active

        table_names: Optional[Sequence[Optional[str]]]
        if table_names := layer.metadata.get("table_names"):
            return table_names  # type: ignore[no-any-return]
        return None

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

        if layer is not None and "adata" in layer.metadata:
            self.model.adata = layer.metadata["adata"]

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

    def __init__(self, viewer: Viewer):
        super().__init__()

        self._viewer = viewer
        self._model = DataModel()

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

        self.model.events.adata.connect(self._on_layer_update)
        self.model.events.color_by.connect(self._change_color_by)

    def _on_layer_update(self, event: Optional[Any] = None) -> None:
        """When the model updates the selected layer, update the relevant widgets."""
        logger.info("Updating layer.")

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
                if isinstance(layer, (Points, Shapes)) and (cols_df := layer.metadata.get("_columns_df")) is not None:
                    self.dataframe_columns_widget.addItems(map(str, cols_df.columns))
                    self.model.system_name = layer.metadata.get("name", None)
            return

        if layer is not None and "adata" in layer.metadata:
            self.model.adata = layer.metadata["adata"]

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

        if layer is not None and "adata" in layer.metadata:
            with self.model.events.adata.blocker():
                self.model.adata = layer.metadata["adata"]

        if self.model.adata.shape == (0, 0):
            return

        self.model.instance_key = layer.metadata["instance_key"] = (
            table.uns["spatialdata_attrs"]["instance_key"] if table is not None else None
        )
        self.model.region_key = layer.metadata["region_key"] = (
            table.uns["spatialdata_attrs"]["region_key"] if table is not None else None
        )
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

    def _get_adata_layer(self) -> Sequence[Optional[str]]:
        adata_layers = list(self.model.adata.layers.keys())
        if len(adata_layers):
            return adata_layers
        return [None]

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
    def layernames(self) -> FrozenSet[str]:
        """Names of :class:`napari.layers.Layer`."""
        return frozenset(layer.name for layer in self.viewer.layers)


class QtAdataAnnotationWidget(QWidget):
    """Adata viewer widget."""

    def __init__(self, input: Viewer):
        super().__init__()
        self._viewer = input

        self.setLayout(QGridLayout())
        self._current_color = "#FFFFFF"
        self._current_class = "undefined"
        self._current_annotator = ""

        # just for testing purposes
        df = pd.DataFrame(
            [
                [1, 9, 2],
                [1, 0, -1],
                [3, 5, 2],
                [3, 3, 2],
                [5, 8, 9],
            ],
            columns=["A", "B", "C"],
            index=["Row 1", "Row 2", "Row 3", "Row 4", "Row 5"],
        )

        self.feature_model = LayerFeatureModel(layer_features_df=df)

        self.annotation_widget = MainWindow()
        self.layout().addWidget(self.annotation_widget)
        self.annotation_widget.table_widget.setModel(self.feature_model)
        self.annotation_widget.button_group.buttonClicked.connect(self._on_click)
        self.viewer.layers.events.inserted.connect(self._on_inserted)
        self.viewer.layers.selection.events.changed.connect(self._on_layer_selection_changed)

    def _on_inserted(self, event):
        layer = event.value
        if isinstance(layer, Shapes):
            if sdata := layer.metadata.get("sdata"):
                self._set_tables(sdata)
            # layer.events.data.connect(self._update_annotations)
            df = pd.DataFrame(
                {
                    "class": pd.Series([], dtype="str"),
                    "color": pd.Series([], dtype="str"),
                    "description": pd.Series([], dtype="str"),
                }
            )
            layer.features = df
            self._viewer.layers.selection.active.current_face_color = self._current_color
            self.feature_model.dataframe = df
            self.annotation_widget.table_widget.setModel(self.feature_model)

    @property
    def viewer(self) -> napari.Viewer:
        """:mod:`napari` viewer."""
        return self._viewer

    def _on_layer_selection_changed(self):
        layer = self._viewer.layers.selection.active
        if isinstance(layer, Shapes):
            if sdata := layer.metadata.get("sdata"):
                self._set_tables(sdata)
            # layer.events.data.connect(self._update_annotations)
            df = pd.DataFrame(
                {
                    "class": pd.Series([], dtype="category"),
                    "color": pd.Series([], dtype="str"),
                    "description": pd.Series([], dtype="str"),
                }
            )
            layer.features = df
            self.feature_model.dataframe = df

    # def _update_annotations(self, event: Event) -> None:
    #     layer = event.source
    #     if event.action == "added" and len(event.data_indices) == 1:
    #         self._update_layer_features(layer, event.action)
    #         self._insert_table_widget_row(event)
    #     elif len(event.data_indices) > 1:
    #         raise ValueError(f"Can only add one annotation at the time, got {len(event.data_indices)}")

    # def _insert_table_widget_row(self, event: Event) -> None:
    #     table = self.annotation_widget.table_widget
    #     table_index = table.rowCount()
    #
    #     # when data is being added by the user the index is always -1
    #     index = event.data_indices[0] if event.data_indices[0] != -1 else len(event.value)
    #
    #     table.insertRow(table_index)
    #     table.setItem(table_index, 0, QTableWidgetItem(str(index)))
    #     table.setItem(table_index, 1, QTableWidgetItem(self._current_class))
    #     table.setItem(table_index, 2, QTableWidgetItem(self._current_annotator))
    #     table.setItem(table_index, 3, QTableWidgetItem(self._current_color))

    # def _remove_table_widget_row(self, event: Event) -> None:

    def _update_layer_features(self, layer, action: str) -> None:
        if action == "added":
            row = [self._current_class, self._current_color]
            layer.features.loc[len(layer.features) - 1] = row

    def _on_click(self):
        if isinstance(self.viewer.layers.selection.active, Shapes):
            # We have five columns with at position 1 the color button
            color_ind = self.annotation_widget.tree_view.selectedIndexes()[1]
            color_button = self.annotation_widget.tree_view.indexWidget(color_ind)

            class_ind = self.annotation_widget.tree_view.selectedIndexes()[2]
            class_widget = self.annotation_widget.tree_view.indexWidget(class_ind)
            self._current_class = class_widget.text()

            # .name() converts the QColor in hexadecimal
            palette = color_button.palette()
            color = palette.color(color_button.backgroundRole()).name()
            self._current_color = color
            self._viewer.layers.selection.active.current_face_color = color

    def _set_tables(self, sdata):
        self.annotation_widget.table_name_widget.clear()
        table_list = [""]
        for table_name, table in sdata.tables.items():
            if "spatialdata_attrs" not in table.uns:
                table_list.append(table_name)

        self.annotation_widget.table_name_widget.addItems(table_list)
