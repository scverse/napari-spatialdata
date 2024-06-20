from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import pandas as pd
import pyqtgraph as pg
from loguru import logger
from napari.viewer import Viewer
from pandas.api.types import CategoricalDtype
from pyqtgraph import GraphicsLayoutWidget
from pyqtgraph.graphicsItems import ROI
from qtpy import QtWidgets
from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtGui import QColor, QIcon
from qtpy.QtWidgets import QPushButton

from napari_spatialdata._model import DataModel
from napari_spatialdata._widgets import AListWidget, ComponentWidget
from napari_spatialdata.utils._utils import NDArrayA

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
        self.text = None
        self.chosen = None

    def _onAction(self, items: Iterable[str]) -> None:
        for item in sorted(set(items)):

            # check if there is an annotation connecting to the main viewer
            # if not, only the main column will be retrieved
            # test!!!

            try:
                self._model.instance_key = item
                logger.info(f"Getting {self.getAttribute()} data for {item} at {self}")
                vec, _ = self._getter(item, index=self.getIndex())
            except Exception as e:  # noqa: BLE001
                logger.error(e)
                logger.info(self)
                continue
            self.chosen = item
            if isinstance(vec, np.ndarray):
                self.data = {"vec": vec}
            elif vec is not None and isinstance(vec.dtype, CategoricalDtype):
                sorted_set = sorted(set(vec))
                category_map = {category: index for index, category in enumerate(sorted_set)}
                self.data = {"vec": np.array([category_map[cat] for cat in vec]), "labels": sorted_set}

                # colortypes = _set_palette(self.model.adata, key=item, palette=self.model.palette, vec=vec)
                # if self._color:
                #     self.data = {
                #         "vec": _get_categorical(
                #             self.model.adata, key=item, vec=vec, palette=self.model.palette, colordict=colortypes
                #         ),
                #         "cat": vec,
                #         "palette": colortypes,
                #     }
            elif vec is None:
                self.data = None
            else:
                raise TypeError(f"The chosen field's datatype ({vec.dtype.name}) cannot be plotted")
        return

    def _onOneClick(self, items: Iterable[str]) -> None:
        if self.getAttribute() == "obsm":
            logger.info(f"{self} should be skipping action")
            return
        logger.info(f"{self} is calling action")
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
                assert isinstance(text, (int, str))
            self.text = text  # type: ignore[assignment]
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


class PlotWidget(GraphicsLayoutWidget):

    def __init__(self, viewer: Viewer | None, model: DataModel):

        self.is_widget = False
        if viewer is None:
            viewer = Viewer()
            self.is_widget = True

        super().__init__()

        self._viewer = viewer
        self._model = model

        if self.is_widget:
            self._viewer.close()

        self.scatter_plot = self.addPlot(title="")

        self.scatter = None
        self.x_data: NDArrayA | pd.Series | None = None
        self.y_data: NDArrayA | pd.Series | None = None
        self.color_data: NDArrayA | pd.Series | None = None
        self.brushes: list[Any] | None = None

        # threshold for adding new vertex to the ROI
        self.vertex_add_threshold = 20

        # initialize shapes info
        self.current_roi: ROI | None = None
        self.last_pos: tuple[Any, Any] | None = None
        self.current_points: list[tuple[Any, Any]] = []
        self.roi_list: list[ROI] = []

        # Adding a button to toggle auto-range
        self.auto_range_button = QPushButton(self)
        self.auto_range_button.setIcon(
            QIcon(r"../../src/napari_spatialdata/resources/icons8-home-48.png")
        )  # Set the icon image
        self.auto_range_button.setIconSize(QSize(24, 24))  # Icon size
        self.auto_range_button.setStyleSheet("QPushButton {background-color: transparent;}")
        self.auto_range_button.clicked.connect(self.use_auto_range)
        self.auto_range_button.setToolTip("Auto Range")
        self.auto_range_button.move(10, 10)

        # Drawing mode toggle button
        self.drawing = False
        self.drawing_mode_button = QPushButton(self)
        self.drawing_mode_button.setIcon(QIcon(r"../../src/napari_spatialdata/resources/icons8-polygon-80.png"))
        self.drawing_mode_button.setIconSize(QSize(24, 24))
        self.drawing_mode_button.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                border: none;
            }
            QPushButton:checked {
                border: 1px solid white;
            }
        """
        )
        self.drawing_mode_button.setCheckable(True)
        self.drawing_mode_button.clicked.connect(self.toggle_drawing_mode)
        self.auto_range_button.setToolTip("Add ROIs.")
        self.drawing_mode_button.move(50, 10)  # Adjust position as needed

        # Connect mouse events
        self.scatter_plot.setMouseEnabled(x=True, y=True)
        self.scatter_plot.scene().sigMouseClicked.connect(self.mousePressEvent)
        self.scatter_plot.scene().sigMouseClicked.connect(self.mouseMoveEvent)

        # add the gradient widget with the histogram
        self.lut = pg.HistogramLUTItem(gradientPosition="right")
        self.lut.disableAutoHistogramRange()
        self.lut.gradient.sigGradientChanged.connect(self.on_gradient_changed)

        # set default colormap with no triangle ticks
        color_map = pg.colormap.get("viridis")
        self.lut.gradient.setColorMap(color_map)
        st = self.lut.gradient.saveState()
        st["ticksVisible"] = False
        self.lut.gradient.restoreState(st)

        # connect the signal to update the scatter plot colors
        self.lut.sigLevelChangeFinished.connect(self.plot)
        self.addItem(self.lut, row=0, col=2)

    def on_gradient_changed(self) -> None:
        """Update the scatter plot colors when the gradient is changed."""
        self.plot()

    def get_brushes(self, event: Any = None) -> list[QColor] | None:
        """Get the brushes for the scatter plot based on the color data."""
        if self.color_data is not None:

            level_min, level_max = self.lut.getLevels()

            data = np.clip(self.color_data, level_min, level_max)
            data = (data - level_min) / (level_max - level_min)

            return [pg.mkBrush(*x) for x in self.lut.gradient.colorMap().map(data)]

        return None

    def toggle_drawing_mode(self) -> None:
        self.drawing = not self.drawing
        if self.drawing:
            self.scatter_plot.setMouseEnabled(x=False, y=False)
            self.scatter_plot.setMenuEnabled(False)
            self.scatter_plot.enableAutoRange("xy", False)
        else:
            self.scatter_plot.setMouseEnabled(x=True, y=True)
            self.scatter_plot.setMenuEnabled(True)
            # self.scatter_plot.enableAutoRange("xy", True)

    def mousePressEvent(self, event: Any) -> None:
        if self.drawing and event.button() == Qt.LeftButton:
            logger.info("Left press detected")

            pen = pg.mkPen(color="gray", width=2)
            hoverPen = pg.mkPen(color="red", width=2)
            handlePen = pg.mkPen(color="red", width=2)

            self.current_roi = pg.PolyLineROI(
                [], closed=True, removable=True, pen=pen, handlePen=handlePen, hoverPen=hoverPen
            )
            self.scatter_plot.addItem(self.current_roi)

            plot_pos = self.scatter_plot.vb.mapSceneToView(event.pos())
            self.current_points = [(plot_pos.x(), plot_pos.y())]
            self.last_pos = (event.pos().x(), event.pos().y())

            event.accept()

        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:
        if self.drawing and self.current_roi is not None:

            if self.last_pos is not None:
                dist = np.sqrt((event.pos().x() - self.last_pos[0]) ** 2 + (event.pos().y() - self.last_pos[1]) ** 2)

                if dist > self.vertex_add_threshold:

                    plot_pos = self.scatter_plot.vb.mapSceneToView(event.pos())
                    self.current_points.append((plot_pos.x(), plot_pos.y()))

                    self.current_roi.setPoints(self.current_points)

                    self.last_pos = (event.pos().x(), event.pos().y())

            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        if self.drawing and event.button() == Qt.LeftButton:

            logger.info("Left button released.")

            if (self.current_roi is not None) and (len(self.current_points) > 2):

                # Connect the signal for removing the ROI
                self.current_roi.sigRemoveRequested.connect(self.remove_roi)

                # store with other ROIs
                self.roi_list.append(self.current_roi)

            self.current_roi = None
            self.current_points = []

            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: Any) -> None:
        if event.button() == Qt.LeftButton:
            plot_pos = self.scatter_plot.vb.mapSceneToView(event.pos())

            for roi in self.roi_list:
                if roi.contains(plot_pos):
                    logger.info(f"Remove {roi}.")
                    self.scatter_plot.removeItem(roi)
                    self.roi_list.remove(roi)
                    break

            event.accept()

    def remove_roi(self, roi: ROI) -> None:
        # Remove the specific ROI that emitted the signal
        logger.info(f"Remove {self.current_roi} ROI.")
        self.scatter_plot.removeItem(roi)
        self.roi_list.remove(roi)

    def use_auto_range(self) -> None:
        """Default display of the graph."""
        if self.color_data is not None:
            color_min = self.color_data.min()
            color_max = self.color_data.max()
        else:
            color_min = 0
            color_max = 1
        self.lut.setLevels(color_min, color_max)
        self.plot()
        self.scatter_plot.enableAutoRange("xy", True)

    def _onClick(
        self,
        x_data: dict[str, Any] | None,
        y_data: dict[str, Any] | None,
        color_data: dict[str, Any] | None,
        x_label: str | None,
        y_label: str | None,
        color_label: str | None,
    ) -> None:
        self.cat = None
        self.palette = None

        x_changed = False
        y_changed = False

        if np.any(x_data != self.x_data):
            x_changed = True
            self.x_data = x_data["vec"] if x_data else None
            self.x_ticks = x_data.get("labels") if x_data else None
            self.x_label = x_label if x_label else "Count"

            self.scatter_plot.setLabel("bottom", self.x_label)

        if np.any(y_data != self.y_data):
            y_changed = True
            self.y_data = y_data["vec"] if y_data else None
            self.y_ticks = y_data.get("labels") if y_data else None
            self.y_label = y_label if y_label else "Count"

            self.scatter_plot.setLabel("left", self.y_label)

        if np.any(color_data != self.color_data):
            self.color_data = color_data["vec"] if color_data else None
            self.color_names = color_data.get("labels") if color_data else None
            self.color_label = color_label

            if self.color_data is not None:
                y, x = np.histogram(self.color_data, density=True, bins=100)
                self.lut.plot.setData(x[:-1], y)
                self.lut.setLevels(np.min(self.color_data), np.max(self.color_data))
                self.lut.autoHistogramRange()
            else:
                self.lut.plot.setData(None, None)
                self.lut.setLevels(0, 1)
                self.lut.disableAutoHistogramRange()

            # generate brushes
            self.brushes = self.get_brushes()

        if x_changed or y_changed:
            self.scatter_plot.getAxis("bottom").setTicks(None)
            self.scatter_plot.getAxis("left").setTicks(None)

        self.plot()

        # rescale for new data
        if x_changed or y_changed:
            self.scatter_plot.enableAutoRange("xy", True)
            self.update_ticks()

    def update_ticks(self) -> None:

        axis = self.scatter_plot.getAxis("bottom")
        if self.x_ticks is not None:
            axis.setTicks([[(i, str(x)) for i, x in enumerate(self.x_ticks)]])
            # axis.setStyle(tickTextAngle=45

        axis = self.scatter_plot.getAxis("left")
        if self.y_ticks is not None:
            axis.setTicks([[(i, str(y)) for i, y in enumerate(self.y_ticks)]])

    def plot(self, event: Any = None) -> None:

        # self.clear_plot()

        if self.x_data is not None or self.y_data is not None:

            logger.info("Generating scatter plot...")

            # plot the scatter plot
            if self.x_data is not None and self.y_data is not None:
                self.scatter = self.scatter_plot.plot(
                    self.x_data, self.y_data, pen=None, symbol="o", symbolBrush=self.brushes, clear=True
                )
            # plot the pseudo scatter plot on x axis
            elif self.x_data is not None:
                ps = pg.pseudoScatter(self.x_data)
                self.scatter = self.scatter_plot.plot(
                    self.x_data, ps, fillLevel=0, pen=None, symbolBrush=self.brushes, clear=True
                )
            # plot the pseudo scatter plot on y axis
            elif self.y_data is not None:
                ps = pg.pseudoScatter(self.y_data)
                self.scatter = self.scatter_plot.plot(
                    ps, self.y_data, fillLevel=0, pen=None, symbolBrush=self.brushes, clear=True
                )

    # def clear_plot(self) -> None:
    #     """Clears the scatter plot"""

    #     if self.scatter:
    #         self.scatter_plot.removeItem(self.scatter)
    #         self.scatter = None
