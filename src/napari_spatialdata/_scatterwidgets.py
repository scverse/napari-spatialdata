from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyqtgraph as pg
from loguru import logger
from napari.qt import get_current_stylesheet
from napari.utils.colormaps import label_colormap
from napari.viewer import Viewer
from pandas.api.types import CategoricalDtype
from pyqtgraph import GraphicsLayoutWidget, GraphicsWidget
from pyqtgraph.graphicsItems import ROI
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.Qt.QtCore import pyqtSignal
from pyqtgraph.widgets.ColorButton import ColorButton
from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtGui import QColor, QIcon
from qtpy.QtWidgets import QPushButton
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon
from spatialdata._types import ArrayLike

from napari_spatialdata._model import DataModel
from napari_spatialdata._widgets import AListWidget, ComponentWidget
from napari_spatialdata.constants.config import POINT_SIZE_SCATTERPLOT_WIDGET

__all__ = [
    "PlotWidget",
    "AxisWidgets",
]


class ScatterListWidget(AListWidget):
    attrChanged = Signal()
    _text = None
    _chosen = None

    def __init__(self, model: DataModel, attr: str, color: bool, **kwargs: Any):
        AListWidget.__init__(self, None, model, attr, **kwargs)
        self.attrChanged.connect(self._onChange)
        self._color = color
        self._data: dict[str, Any] | None = None
        self.itemClicked.connect(lambda item: self._onOneClick((item.text(),)))

    def _onChange(self) -> None:
        AListWidget._onChange(self)
        self.data = None
        self.text = None
        self.chosen = None

    def _onAction(self, items: Iterable[str]) -> None:

        for item in sorted(set(items)):
            try:
                vec, _, _ = self._getter(item, index=self.getIndex())
            except Exception as e:  # noqa: BLE001
                logger.error(e)
                logger.info(self)
                continue
            self.chosen = item

            if isinstance(vec, np.ndarray):
                self.data = {"vec": vec}
            elif vec is not None and isinstance(vec.dtype, (CategoricalDtype | bool)):
                try:
                    sorted_set = sorted(set(vec), key=int)
                except ValueError:
                    sorted_set = sorted(set(vec))
                category_map = {category: index for index, category in enumerate(sorted_set)}
                self.data = {"vec": np.array([category_map[cat] for cat in vec]), "labels": sorted_set}

            elif vec is None:
                self.data = None
            else:
                raise TypeError(f"The chosen field's datatype ({vec.dtype.name}) cannot be plotted")
        return

    def _onOneClick(self, items: Iterable[str]) -> None:
        if self.getAttribute() == "obsm":
            return
        self._onAction(items)
        return

    def setAttribute(self, field: str | None) -> None:
        if field == self.getAttribute():
            return
        if field not in ("None", "var", "obs", "obsm"):
            raise ValueError(f"{field} is not a valid adata field.")
        if field == "None":
            self.data = None
            self._attr = "None"
            self._getter = lambda: None
        else:
            self._attr = field
            self._getter = getattr(self.model, f"get_{field}")
        self.attrChanged.emit()

    def getAttribute(self) -> str | None:
        if TYPE_CHECKING:
            assert isinstance(self._attr, str)
        return self._attr

    def setComponent(self, text: int | str | None) -> None:

        if self.getAttribute() == "var":
            if TYPE_CHECKING:
                assert isinstance(text, str)
            self.text = text
            super().setAdataLayer(text)
        elif self.getAttribute() == "obsm":
            if TYPE_CHECKING:
                assert isinstance(text, int | str)
            self.text = text
            super().setIndex(text)

    @property
    def text(self) -> str | None:
        return self._text

    @text.setter
    def text(self, text: str | int | None) -> None:
        self._text = text

    @property
    def chosen(self) -> str | None:
        return self._chosen

    @chosen.setter
    def chosen(self, chosen: str | None) -> None:
        self._chosen = chosen

    @property
    def data(self) -> None | dict[str, Any] | None:
        return self._data

    @data.setter
    def data(self, data: None | dict[str, Any]) -> None:
        self._data = data


class AxisWidgets(QtWidgets.QWidget):
    def __init__(self, model: DataModel, name: str, color: bool = False):
        super().__init__()

        self._model = model

        selection_label = QtWidgets.QLabel(f"{name} type:")
        selection_label.setToolTip("Select between obs, obsm and var.")

        self.selection_widget = QtWidgets.QComboBox()
        self.selection_widget.addItem("None", None)
        self.selection_widget.addItem("obsm", None)
        self.selection_widget.addItem("obs", None)
        self.selection_widget.addItem("var", None)

        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().addWidget(selection_label)
        self.layout().addWidget(self.selection_widget)

        label = QtWidgets.QLabel(f"Select for {name}:")
        label.setToolTip(f"Select {name}.")

        self.widget = ScatterListWidget(self.model, attr="None", color=color)
        self.widget.setAttribute("None")

        self.component_widget = ComponentWidget(self.model, attr="None")
        self.component_widget.setToolTip("Select axis of obsm data.")
        self.component_widget.currentTextChanged.connect(self.widget.setComponent)
        self.widget.itemClicked.connect(self.component_widget._onClickChange)

        self.layout().addWidget(label)
        self.layout().addWidget(self.widget)
        self.layout().addWidget(self.component_widget)

        self.selection_widget.currentTextChanged.connect(self.widget.setAttribute)
        self.selection_widget.currentTextChanged.connect(self.component_widget.setAttribute)
        self.selection_widget.currentTextChanged.connect(self.component_widget.setToolTip)

    def getFormattedLabel(self) -> str | None:
        return (
            str(self.widget.getAttribute()) + ": " + str(self.widget.chosen)
            if self.widget.text is None
            else (
                str(self.widget.getAttribute()) + ": " + str(self.widget.chosen) + "[" + str(self.widget.text) + "]"
                if self.widget.getAttribute() == "obsm"
                else str(self.widget.getAttribute())
                + ": "
                + str(self.widget.chosen)
                + " on layer "
                + str(self.widget.text)
            )
        )

    def clear(self) -> None:
        self.widget.clear()
        self.component_widget.clear()

    @property
    def model(self) -> DataModel:
        """:mod:`napari` viewer."""
        return self._model


class DiscreteColorWidget(GraphicsWidget):

    # Define the custom signal
    paletteChanged = pyqtSignal()

    def __init__(self, model: DataModel, color_data: dict[str, Any]):
        super().__init__()

        self._model = model

        self.layout = QtWidgets.QGraphicsLinearLayout(QtCore.Qt.Vertical)
        self.layout.setContentsMargins(0, 0, 0, 0)  # Set minimal margins
        self.layout.setSpacing(0)  # Set minimal spacing
        self.setLayout(self.layout)

        self.color_buttons = {}

        # TODO- format of the palette in the model itself
        # if self._model.palette is not None:
        #     self.palette = self._model.palette
        # else:
        self.palette = label_colormap(len(color_data["labels"]))

        for ind, obj_category in enumerate(color_data["labels"]):

            obj_category = str(obj_category)

            h_layout = QtWidgets.QGraphicsLinearLayout(QtCore.Qt.Horizontal)
            h_layout.setContentsMargins(0, 0, 0, 0)

            label_widget = GraphicsWidget()
            label_layout = QtWidgets.QGraphicsLinearLayout(QtCore.Qt.Horizontal)
            label_layout.setContentsMargins(0, 0, 0, 0)

            text_proxy = QtWidgets.QGraphicsProxyWidget()
            label = QtWidgets.QLabel(obj_category)
            label.setStyleSheet("background-color: black; color: white;")
            text_proxy.setWidget(label)

            label_layout.addItem(text_proxy)
            label_widget.setLayout(label_layout)

            color = self.palette.map(ind + 1) * 255
            color_button = ColorButton(color=color)
            color_button.setMinimumSize(60, 30)
            color_button.setMaximumSize(60, 30)
            color_button.setStyleSheet(
                """
            QPushButton {
                background-color: transparent;
                border: none;
            }
            """
            )

            self.color_buttons[obj_category] = color_button

            # connect to the color change
            color_button.sigColorChanged.connect(self.on_color_changed)

            button_proxy = QtWidgets.QGraphicsProxyWidget()
            button_proxy.setWidget(color_button)

            h_layout.addItem(button_proxy)
            h_layout.addItem(label_widget)
            self.layout.addItem(h_layout)

    def on_color_changed(self, color_button: ColorButton) -> None:
        for key, value in self.color_buttons.items():
            if value is color_button:
                obj_category = key
                break
        logger.info(f"Color changed for {obj_category}.")
        new_color = color_button.color().getRgbF()
        index = list(self.color_buttons.keys()).index(obj_category)
        self.palette.colors[index + 1] = new_color

        # Emit the signal
        self.paletteChanged.emit()


class HoverScatterPlotItem(pg.PlotDataItem):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.setAcceptHoverEvents(True)

    def hoverEvent(self, event: Any) -> None:
        widget = self.getViewWidget()
        if event.isExit():
            widget.clear_hover_highlight()
        else:
            scene_pos = event.scenePos()
            plot_pos = self.mapToParent(scene_pos)
            widget.update_hover_highlight(plot_pos.x(), plot_pos.y())


class PlotWidget(GraphicsLayoutWidget):

    def __init__(self, viewer: Viewer | None, model: DataModel) -> None:

        self.is_widget = False
        if viewer is None:
            viewer = Viewer()
            self.is_widget = True

        self._init_complete = False

        super().__init__()

        self._viewer = viewer
        self._model = model

        if self.is_widget:
            self._viewer.close()

        self.scatter: HoverScatterPlotItem | None = None
        self.x_data: ArrayLike | pd.Series | None = None
        self.y_data: ArrayLike | pd.Series | None = None
        self.kd_tree: cKDTree | None = None
        self.dist_threshold = [None, None]
        self.color_data: dict[str, Any] | None = None
        self.color_label: str | None = None
        self.x_label: str | None = "None"
        self.y_label: str | None = "None"
        self.brushes: list[Any] | None = None
        self.lut: pg.HistogramLUTItem | None = None
        self.discrete_color_widget: DiscreteColorWidget | None = None
        self.wrapped_widget: QtWidgets.QGraphicsProxyWidget | None = None

        self.scatter_plot = self.addPlot(title="")
        self.scatter_plot.setLabel("bottom", self.x_label)
        self.scatter_plot.setLabel("left", self.x_label)
        self.scatter_plot.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # Connect the view range change signal to the update function
        self.scatter_plot.sigRangeChanged.connect(self.on_zoom_event)

        # Create a separate ScatterPlotItem for the highlighted point
        self.hovered_point = pg.ScatterPlotItem(
            pen=pg.mkPen("r", width=2), symbol="o", size=POINT_SIZE_SCATTERPLOT_WIDGET, brush=pg.mkBrush(255, 0, 0)
        )
        self.scatter_plot.addItem(self.hovered_point)

        # get default color of the scatter plot
        self.color = tuple([x * 255 for x in self.scatter_plot.getAxis("bottom").pen().color().getRgbF()[:3]])

        # threshold for adding new vertex to the ROI
        self.vertex_add_threshold = 20

        # Setup a timer for debouncing
        self.debounce_timer = QtCore.QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self.on_debounced_zoom_event)

        # initialize shapes info
        self.current_roi: ROI | None = None
        self.last_pos: tuple[Any, Any] | None = None
        self.current_points: list[tuple[Any, Any]] = []
        self.roi_list: list[ROI] = []
        self.initial_pos: tuple[Any, Any] | None = None

        # Adding a button for a default view
        self.auto_range_button = QPushButton(self)
        self.auto_range_button.setIcon(QIcon(str(Path(__file__).parent / "resources/icons8-home-48.png")))
        self.auto_range_button.setIconSize(QSize(24, 24))  # Icon size
        self.auto_range_button.setStyleSheet("QPushButton {background-color: transparent;}")
        self.auto_range_button.clicked.connect(self.use_auto_range)
        self.auto_range_button.setToolTip("Auto Range")
        self.auto_range_button.move(10, 10)

        # Polygon drawing mode toggle button
        self.drawing = False
        self.drawing_mode_button = QPushButton(self)
        self.drawing_mode_button.setIcon(QIcon(str(Path(__file__).parent / "resources/icons8-polygon-80.png")))
        self.drawing_mode_button.setIconSize(QSize(24, 24))
        self.drawing_mode_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: transparent;
                border: none;
            }}
            QPushButton:checked {{
                border: 1px solid rgb{self.color};
            }}
        """
        )
        self.drawing_mode_button.setCheckable(True)
        self.drawing_mode_button.clicked.connect(self.toggle_drawing_mode)
        self.drawing_mode_button.setToolTip("Add freehand ROIs.")
        self.drawing_mode_button.move(90, 10)  # Adjust position as needed

        # Rectangle drawing mode toggle button
        self.rectangle = False
        self.rectangle_mode_button = QPushButton(self)
        self.rectangle_mode_button.setIcon(QIcon(str(Path(__file__).parent / "resources/icons8-rectangle-48.png")))
        self.rectangle_mode_button.setIconSize(QSize(24, 24))
        self.rectangle_mode_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: transparent;
                border: none;
            }}
            QPushButton:checked {{
                border: 1px solid rgb{self.color};
            }}
        """
        )
        self.rectangle_mode_button.setCheckable(True)
        self.rectangle_mode_button.clicked.connect(self.toggle_rectangle_mode)
        self.rectangle_mode_button.setToolTip("Add rectangular ROIs.")
        self.rectangle_mode_button.move(50, 10)  # Adjust position as needed

        # Connect mouse events
        self.scatter_plot.setMouseEnabled(x=True, y=True)
        self.scatter_plot.scene().sigMouseMoved.connect(self.mouseMoveEvent)
        self.scatter_plot.scene().sigMouseClicked.connect(self.mousePressEvent)
        self.scatter_plot.scene().sigMouseClicked.connect(self.mouseReleaseEvent)

        # Add labels for cursor position and data point value
        self.cursor_position_label = QtWidgets.QLabel(self)
        self.cursor_position_label.setStyleSheet(f"QLabel {{ background-color : black; color : rgb{self.color}; }}")
        self.cursor_position_label.setFixedSize(150, 20)

        self.data_point_label = QtWidgets.QLabel(self)
        self.data_point_label.setStyleSheet(f"QLabel {{ background-color : black; color : rgb{self.color}; }}")
        self.data_point_label.setFixedSize(150, 20)

        self._init_complete = True
        self.update_label_positions()

    def on_zoom_event(self) -> None:
        # Start or restart the debounce timer with a 100ms delay
        self.debounce_timer.start(100)

    def on_debounced_zoom_event(self) -> None:
        # Update the proximity sensitivity
        self.update_proximity_sensitivity()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Resizing of the window."""
        super().resizeEvent(event)
        if self._init_complete:
            self.update_label_positions()

    def update_label_positions(self) -> None:
        """Update the position of the cursor position and data point labels."""
        widget_height = self.size().height()
        self.cursor_position_label.move(30, widget_height - 25)
        self.data_point_label.move(170, widget_height - 25)

    def update_proximity_sensitivity(self) -> None:
        """Update the proximity sensitivity of the hover highlight."""
        if self.scatter is not None:
            logger.debug("Updating proximity sensitivity...")
            self.dist_threshold = [x * 15 for x in self.scatter.pixelSize()]  # Adjust this factor as needed

    def update_hover_highlight(self, x: float, y: float) -> None:
        """Update the hover highlight based on the cursor position."""
        if self.kd_tree is not None and self.x_data is not None and self.y_data is not None:
            # Query the k-d tree for the nearest neighbor
            dist, idx = self.kd_tree.query([x, y], k=1)
            dist_x = abs(x - self.x_data[idx])
            dist_y = abs(y - self.y_data[idx])
            if dist_x < self.dist_threshold[0] and dist_y < self.dist_threshold[1]:
                self.hovered_point.setData([self.x_data[idx]], [self.y_data[idx]])
                self.hovered_point.setZValue(10)
                value = self.color_vec[idx] if self.color_vec is not None else "N/A"
                if self.color_names is not None:
                    value = self.color_names[value]
                self.data_point_label.setText(f"Value: {value}")
            else:
                self.hovered_point.setData([], [])
                self.data_point_label.setText("Value: N/A")
        else:
            self.hovered_point.setData([], [])
            self.data_point_label.setText("Value: N/A")

    def clear_hover_highlight(self) -> None:
        """Clear the hover highlight."""
        self.hovered_point.setData([], [])
        self.data_point_label.setText("Value: N/A")

    def use_auto_range(self) -> None:
        """Default display of the graph."""
        if self.lut is not None:
            assert self.color_vec is not None
            color_min = self.color_vec.min()
            color_max = self.color_vec.max()

            self.lut.setLevels(color_min, color_max)

        elif self.color_data and self.discrete_color_widget is not None:

            # rebuild discrete color widget
            self.removeItem(self.wrapped_widget)

            self.discrete_color_widget = DiscreteColorWidget(self._model, self.color_data)
            self.wrapped_widget = self.wrap_discrete_color_widget()
            self.addItem(self.wrapped_widget, row=0, col=2)

            # connect the signal to update the scatter plot colors
            self.discrete_color_widget.paletteChanged.connect(self.on_gradient_changed)

        self.brushes = self.get_brushes()
        self.plot()
        self.scatter_plot.enableAutoRange("xy", True)

        # reset ROI modes
        self.switch_to_default_mode()

    def toggle_rectangle_mode(self) -> None:
        self.rectangle = not self.rectangle
        if self.rectangle:
            self.drawing = False
            self.drawing_mode_button.setChecked(False)
            self._enable_rectangle_mode()
        else:
            self._disable_rectangle_mode()

    def toggle_drawing_mode(self) -> None:
        self.drawing = not self.drawing
        if self.drawing:
            self.rectangle = False
            self.rectangle_mode_button.setChecked(False)
            self._enable_drawing_mode()
        else:
            self._disable_drawing_mode()

    def _update_scatter_plot(self, mouse_enabled: bool, menu_enabled: bool, auto_range_enabled: bool) -> None:
        self.scatter_plot.setMouseEnabled(x=mouse_enabled, y=mouse_enabled)
        self.scatter_plot.setMenuEnabled(menu_enabled)
        self.scatter_plot.enableAutoRange("xy", auto_range_enabled)

    def _enable_drawing_mode(self) -> None:
        self._update_scatter_plot(mouse_enabled=False, menu_enabled=False, auto_range_enabled=False)

    def _disable_drawing_mode(self) -> None:
        self._update_scatter_plot(mouse_enabled=True, menu_enabled=True, auto_range_enabled=True)

    def _enable_rectangle_mode(self) -> None:
        self._update_scatter_plot(mouse_enabled=False, menu_enabled=False, auto_range_enabled=False)

    def _disable_rectangle_mode(self) -> None:
        self._update_scatter_plot(mouse_enabled=True, menu_enabled=True, auto_range_enabled=True)

    def switch_to_default_mode(self) -> None:
        self.drawing = False
        self.drawing_mode_button.setChecked(False)
        self.rectangle = False
        self.rectangle_mode_button.setChecked(False)

        self.current_roi = None
        self.current_points = []
        self.initial_pos = None

        self._disable_rectangle_mode()
        self._disable_drawing_mode()

    def get_selection(self) -> ArrayLike | None:
        """Get the selection from the scatter plot."""

        if self.scatter is None:

            return None

        boolean_vector = np.zeros(len(self.scatter.xData), dtype=bool)

        polygon_list = self.rois_to_polygons()

        for i, (x, y) in enumerate(zip(self.scatter.xData, self.scatter.yData, strict=True)):
            point = Point(x, y)
            # Check if the point belongs to any ROI
            for polygon in polygon_list:

                if polygon.contains(point):
                    boolean_vector[i] = True
                    break

        return boolean_vector

    def rois_to_polygons(self) -> list[Polygon]:
        """Convert ROIs to Shapely Polygons."""

        # create a list of polygons
        polygon_list = []

        for roi in self.roi_list:

            if isinstance(roi, pg.graphicsItems.ROI.PolyLineROI):

                center_point = roi.getState()["pos"]
                polygon_points = [center_point + x for x in roi.getState()["points"]]
                polygon = Polygon(polygon_points)

            elif isinstance(roi, pg.graphicsItems.ROI.RectROI):

                # Get the position and size of the RectROI
                pos = roi.pos()
                size = roi.size()

                # Calculate the vertices of the RectROI
                x0, y0 = pos[0], pos[1]
                x1, y1 = x0 + size[0], y0
                x2, y2 = x1, y1 + size[1]
                x3, y3 = x0, y0 + size[1]

                # Create a list of vertices
                vertices = [(x0, y0), (x1, y1), (x2, y2), (x3, y3), (x0, y0)]

                # Create a Shapely Polygon
                polygon = Polygon(vertices)

            else:
                raise TypeError("Only PolyLineROI and RectROI are supported.")

            polygon_list.append(polygon)

        return polygon_list

    def _onClick(
        self,
        x_data: dict[str, Any] | None,
        y_data: dict[str, Any] | None,
        color_data: dict[str, Any] | None,
        x_label: str | None,
        y_label: str | None,
        color_label: str | None,
    ) -> None:

        logger.debug("Plotting has been requested.")

        self.cat = None
        self.palette = None

        x_changed = False
        y_changed = False

        if x_label != self.x_label:
            x_changed = True
            self.x_data = x_data["vec"] if x_data else None
            self.x_ticks = x_data.get("labels") if x_data else None
            self.x_label = x_label if x_label != "None: None" else "Count"

            self.scatter_plot.setLabel("bottom", self.x_label)

        if y_label != self.y_label:
            y_changed = True
            self.y_data = y_data["vec"] if y_data else None
            self.y_ticks = y_data.get("labels") if y_data else None
            self.y_label = y_label if y_label != "None: None" else "Count"

            self.scatter_plot.setLabel("left", self.y_label)

        # both x and y data are set to None
        if self.x_label == "Count" and self.y_label == "Count":
            self.x_label = None
            self.y_label = None
            color_label = None
            color_data = None

        if color_label != self.color_label:

            logger.debug("Change in color label detected.")

            self.color_data = color_data
            self.color_vec = color_data["vec"] if color_data else None
            self.color_names = color_data.get("labels") if color_data else None
            self.color_label = color_label.replace("__change__", "") if color_label else None

            # clear previous color widgets
            if self.lut:
                self.removeItem(self.lut)
                self.lut = None
            if self.wrapped_widget:
                self.removeItem(self.wrapped_widget)
                self.discrete_color_widget = None
                self.wrapped_widget = None

            # build a new appropriate color widget
            # generate brushes
            if self.color_data is not None:

                # discrete color data
                if self.color_names is not None:

                    self.discrete_color_widget = DiscreteColorWidget(self._model, self.color_data)
                    self.wrapped_widget = self.wrap_discrete_color_widget()
                    self.addItem(self.wrapped_widget, row=0, col=2)

                    # connect the signal to update the scatter plot colors
                    self.discrete_color_widget.paletteChanged.connect(self.on_gradient_changed)

                else:

                    self.lut = self.create_lut_hist()
                    self.addItem(self.lut, row=0, col=2)

                    # connect the signal to update the scatter plot colors
                    self.lut.sigLevelChangeFinished.connect(self.on_gradient_changed)
                    self.lut.gradient.sigGradientChanged.connect(self.on_gradient_changed)

                # generate brushes
                self.brushes = self.get_brushes()
                self.symbolPen = pg.mkPen(color="black", width=1)

            else:

                self.brushes = None
                self.symbolPen = pg.mkPen(color="white", width=1)

        if x_changed or y_changed:
            logger.debug("A change in x or y data has been detected.")
            self.scatter_plot.getAxis("bottom").setTicks(None)
            self.scatter_plot.getAxis("left").setTicks(None)

            # clear ROIs
            for roi in self.roi_list:
                self.scatter_plot.removeItem(roi)
            self.roi_list = []

        self.plot()

        # rescale for new data
        if x_changed or y_changed:
            self.scatter_plot.enableAutoRange("xy", True)
            self.update_ticks()

            # build the KD tree
            if self.x_data is not None and self.y_data is not None:
                self.kd_tree = cKDTree(np.column_stack((self.x_data, self.y_data)))
            else:
                self.kd_tree = None

            # Initial call to set the correct size
            self.update_proximity_sensitivity()

    def plot(self, event: Any = None) -> None:
        """Plot the scatter plot or pseudo histogram."""

        # clear previous
        self.scatter_plot.removeItem(self.scatter)

        if self.x_data is not None or self.y_data is not None:

            logger.debug("Generating scatter plot...")

            # plot the scatter plot
            if self.x_data is not None and self.y_data is not None:
                self.scatter = HoverScatterPlotItem(
                    x=self.x_data,
                    y=self.y_data,
                    pen=None,
                    symbolPen=self.symbolPen,
                    symbol="o",
                    clear=True,
                    symbolBrush=self.brushes,
                )
                self.scatter_plot.addItem(self.scatter)

            # plot the pseudo scatter plot on x axis
            elif self.x_data is not None:
                ps = pg.pseudoScatter(self.x_data)
                self.scatter = self.scatter_plot.plot(
                    self.x_data,
                    ps,
                    fillLevel=0,
                    pen=None,
                    symbolPen=self.symbolPen,
                    symbolBrush=self.brushes,
                    clear=True,
                )
            # plot the pseudo scatter plot on y axis
            elif self.y_data is not None:
                ps = pg.pseudoScatter(self.y_data)
                self.scatter = self.scatter_plot.plot(
                    ps,
                    self.y_data,
                    fillLevel=0,
                    pen=None,
                    symbolPen=self.symbolPen,
                    symbolBrush=self.brushes,
                    clear=True,
                )

            for roi in self.roi_list:
                self.scatter_plot.addItem(roi)

    def create_lut_hist(self) -> pg.HistogramLUTItem:

        # add the gradient widget with the histogram
        lut = pg.HistogramLUTItem(gradientPosition="right")
        lut.disableAutoHistogramRange()

        # set default colormap with no triangle ticks
        color_map = pg.colormap.get("viridis")
        lut.gradient.setColorMap(color_map)
        st = lut.gradient.saveState()
        st["ticksVisible"] = False
        lut.gradient.restoreState(st)

        assert self.color_vec is not None
        y, x = np.histogram(self.color_vec, bins="auto")
        lut.plot.setData(
            np.concatenate([[np.min(self.color_vec)], (x[:-1] + x[1:]) / 2, [np.max(self.color_vec)]]),
            np.concatenate([[0], y, [0]]),
        )
        lut.setLevels(np.min(self.color_vec), np.max(self.color_vec))
        lut.autoHistogramRange()

        return lut

    def on_gradient_changed(self) -> None:
        """Update the scatter plot colors when the gradient is changed."""
        self.brushes = self.get_brushes()
        self.plot()

    def wrap_discrete_color_widget(self) -> QtWidgets.QGraphicsProxyWidget:
        """Wrap the discrete color widget in a GraphicsWidget to make it scrollable."""

        # Create a QGraphicsScene and add the custom widget
        scene = QtWidgets.QGraphicsScene()
        scene.addItem(self.discrete_color_widget)

        # Create a QGraphicsView, set the scene, and make it scrollable
        view = QtWidgets.QGraphicsView(scene)
        view.setRenderHint(QtGui.QPainter.Antialiasing)
        # view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        # sets stle of the vertical slider
        view.setStyleSheet(get_current_stylesheet())
        view.setStyleSheet(
            """
            QGraphicsView {
            border: none;
            background: rgb(0, 0, 0);
            }
            """
        )

        view.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)

        view_proxy = QtWidgets.QGraphicsProxyWidget()
        view_proxy.setWidget(view)

        return view_proxy

    def update_ticks(self) -> None:
        """Update the ticks on the scatter plot axes."""

        axis = self.scatter_plot.getAxis("bottom")
        if self.x_ticks is not None:
            axis.setTicks([[(i, str(x)) for i, x in enumerate(self.x_ticks)]])
            # axis.setStyle(tickTextAngle=45

        axis = self.scatter_plot.getAxis("left")
        if self.y_ticks is not None:
            axis.setTicks([[(i, str(y)) for i, y in enumerate(self.y_ticks)]])

    def get_brushes(self, event: Any = None) -> list[QColor] | None:
        """Get the brushes for the scatter plot based on the color data."""

        logger.debug("Generating brushes...")

        # for discrete data
        if self.discrete_color_widget is not None:

            assert self.color_vec is not None
            return [pg.mkBrush(*x) for x in self.discrete_color_widget.palette.map(self.color_vec + 1) * 255]

        # for continuos data
        if self.lut is not None:

            level_min, level_max = self.lut.getLevels()

            assert self.color_vec is not None
            data = np.clip(self.color_vec, level_min, level_max)
            data = (data - level_min) / (level_max - level_min)

            return [pg.mkBrush(*x) for x in self.lut.gradient.colorMap().map(data)]

        return None

    def mousePressEvent(self, event: Any) -> None:

        if (self.drawing or self.rectangle) and event.button() == Qt.LeftButton:
            logger.debug("Left press detected")

            # Get the position of the click in the scene
            scene_pos = event.pos()

            # Check if the click is within the scatter plot view box
            if not self.scatter_plot.vb.sceneBoundingRect().contains(scene_pos):
                logger.debug("Click outside scatter plot view box, scene position: {scene_pos}")
                return

            pen = pg.mkPen(color="gray", width=2)
            hoverPen = pg.mkPen(color="red", width=2)
            handlePen = pg.mkPen(color="red", width=2)

            self.scatter_plot.enableAutoRange("xy", False)

            if self.drawing:

                plot_pos = self.scatter_plot.vb.mapSceneToView(event.pos())

                self.current_roi = pg.PolyLineROI(
                    [], closed=True, removable=True, pen=pen, handlePen=handlePen, hoverPen=hoverPen
                )
                self.scatter_plot.addItem(self.current_roi)

                self.current_points = [(plot_pos.x(), plot_pos.y())]
                self.last_pos = (event.pos().x(), event.pos().y())

                event.accept()

            if self.rectangle:

                logger.info("Rectangle")

                plot_pos = self.scatter_plot.vb.mapSceneToView(event.pos())
                self.current_roi = pg.RectROI(
                    pos=(plot_pos.x(), plot_pos.y()),
                    size=(0, 0),
                    removable=True,
                    pen=pen,
                    handlePen=handlePen,
                    hoverPen=hoverPen,
                )
                self.scatter_plot.addItem(self.current_roi)

                self.initial_pos = plot_pos

                event.accept()

        else:
            if not isinstance(event, QtGui.QMouseEvent):
                return
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:

        if self.drawing and self.current_roi is not None:

            if self.last_pos is not None:
                dist = np.sqrt((event.pos().x() - self.last_pos[0]) ** 2 + (event.pos().y() - self.last_pos[1]) ** 2)

                if dist > self.vertex_add_threshold:

                    plot_pos = self.scatter_plot.vb.mapSceneToView(event.pos())
                    self.current_points.append((plot_pos.x(), plot_pos.y()))

                    self.current_roi.setPoints(self.current_points)

                    self.last_pos = (event.pos().x(), event.pos().y())

                    logger.debug(f"Updated points for PolyLineROI: {self.current_points}")

            event.accept()

        elif self.rectangle and self.current_roi is not None and self.initial_pos is not None:

            plot_pos = self.scatter_plot.vb.mapSceneToView(event.pos())
            width = plot_pos.x() - self.initial_pos.x()  # type: ignore[attr-defined]
            height = plot_pos.y() - self.initial_pos.y()  # type: ignore[attr-defined]
            self.current_roi.setSize([width, height])

            event.accept()

        else:

            if not isinstance(event, QtGui.QMouseEvent):
                return
            super().mouseMoveEvent(event)

        # Get cursor position
        scene_pos = event.pos()
        plot_pos = self.scatter_plot.vb.mapSceneToView(scene_pos)

        # Update the cursor position label
        self.cursor_position_label.setText(f"X: {plot_pos.x():.2f}, Y: {plot_pos.y():.2f}")

        # Call the hover highlight update function
        self.update_hover_highlight(plot_pos.x(), plot_pos.y())

    def mouseReleaseEvent(self, event: Any) -> None:

        logger.debug(f"Mouse release event detected at {self.scatter_plot.vb.mapSceneToView(event.pos())}")

        if self.drawing and event.button() == Qt.LeftButton:

            if (self.current_roi is not None) and (len(self.current_points) > 2):
                self.current_roi.sigRemoveRequested.connect(self.remove_roi)
                self.roi_list.append(self.current_roi)

            self.switch_to_default_mode()
            event.accept()

        elif self.rectangle and event.button() == Qt.LeftButton:

            if self.current_roi is not None:
                self.current_roi.sigRemoveRequested.connect(self.remove_roi)
                self.roi_list.append(self.current_roi)

            self.switch_to_default_mode()
            event.accept()

        else:
            if not isinstance(event, QtGui.QMouseEvent):
                return
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: Any) -> None:
        if event.button() == Qt.LeftButton:

            plot_pos = self.scatter_plot.vb.mapSceneToView(event.pos())
            point = Point(plot_pos.x(), plot_pos.y())

            self.remove_roi(point=point)

            event.accept()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Handle key press events."""
        if event.key() == Qt.Key_D:
            modifiers = QtWidgets.QApplication.keyboardModifiers()
            if modifiers == Qt.ShiftModifier:
                self.remove_all_rois()
            else:
                self.remove_hovered_roi()
        else:
            super().keyPressEvent(event)

    def remove_all_rois(self) -> None:
        """Remove all ROIs."""
        logger.debug("Removing all ROIs.")
        for roi in self.roi_list:
            self.scatter_plot.removeItem(roi)
        self.roi_list = []

    def remove_hovered_roi(self) -> None:
        """Remove the ROI under the cursor."""

        for roi in self.roi_list:
            if roi.mouseHovering:
                self.remove_roi(roi=roi)
                break

    def remove_roi(self, roi: ROI | None = None, point: Point | None = None) -> None:
        """Remove specified ROI."""

        # find roi by point
        if point is not None:

            polygon_list = self.rois_to_polygons()

            for polygon in polygon_list:

                if polygon.contains(point):

                    roi = self.roi_list[polygon_list.index(polygon)]
                    break

        if roi is not None:
            logger.debug(f"Remove {roi} ROI.")
            self.scatter_plot.removeItem(roi)
            self.roi_list.remove(roi)
