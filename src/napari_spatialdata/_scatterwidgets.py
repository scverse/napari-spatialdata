from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

import matplotlib as plt
import numpy as np
import pandas as pd
import pyqtgraph as pg
from loguru import logger
from napari.viewer import Viewer
from pandas.api.types import CategoricalDtype
from pyqtgraph import GraphicsLayoutWidget
from qtpy import QtWidgets
from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtGui import QColor, QIcon
from qtpy.QtWidgets import QPushButton

from napari_spatialdata._model import DataModel
from napari_spatialdata._widgets import AListWidget, ComponentWidget
from napari_spatialdata.utils._utils import NDArrayA, _get_categorical, _set_palette

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
        self._data: NDArrayA | dict[str, Any] | None = None
        self.itemClicked.connect(lambda item: self._onOneClick((item.text(),)))

    def _onChange(self) -> None:
        AListWidget._onChange(self)
        self.data = None
        self.text = None
        self.chosen = None

    def _onAction(self, items: Iterable[str]) -> None:
        for item in sorted(set(items)):
            try:
                vec, _ = self._getter(item, index=self.getIndex())
            except Exception as e:  # noqa: BLE001
                logger.error(e)
                continue
            self.chosen = item
            if isinstance(vec, np.ndarray):
                self.data = vec
            elif vec is not None and isinstance(vec.dtype, CategoricalDtype):
                self.data = vec
                colortypes = _set_palette(self.model.adata, key=item, palette=self.model.palette, vec=vec)
                if self._color:
                    self.data = {
                        "vec": _get_categorical(
                            self.model.adata, key=item, vec=vec, palette=self.model.palette, colordict=colortypes
                        ),
                        "cat": vec,
                        "palette": colortypes,
                    }
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
        if field not in ("var", "obs", "obsm"):
            raise ValueError(f"{field} is not a valid adata field.")
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
        self._text = text if text is not None else None

    @property
    def chosen(self) -> str | None:
        return self._chosen

    @chosen.setter
    def chosen(self, chosen: str | None) -> None:
        self._chosen = chosen if chosen is not None else None

    @property
    def data(self) -> NDArrayA | dict[str, Any] | None:
        return self._data

    @data.setter
    def data(self, data: NDArrayA | dict[str, Any]) -> None:
        self._data = data


class PlotWidget(GraphicsLayoutWidget):

    def __init__(self, viewer: Viewer | None, model: DataModel):

        self.is_widget = False
        if viewer is None:
            viewer = Viewer()
            self.is_widget = True

        super().__init__()

        self._viewer = viewer
        self._model = model
        self.colorbar = None
        self.selector = None

        if self.is_widget:
            self._viewer.close()

        self.scatter_plot = self.addPlot(title="")
        self.scatter = None

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

        # Handling drawing mode

        # Drawing mode toggle button
        self.drawing = False
        self.drawing_mode_button = QPushButton(self)
        self.drawing_mode_button.setIcon(QIcon(r"../../src/napari_spatialdata/resources/icons8-pencil-drawing-50.png"))
        self.drawing_mode_button.setIconSize(QSize(24, 24))
        self.auto_range_button.setStyleSheet("QPushButton {background-color: transparent;}")
        self.drawing_mode_button.setCheckable(True)
        self.drawing_mode_button.clicked.connect(self.toggle_drawing_mode)
        self.drawing_mode_button.move(50, 10)  # Adjust position as needed

        # Connect mouse events
        self.scatter_plot.setMouseEnabled(x=True, y=True)
        self.scatter_plot.scene().sigMouseClicked.connect(self.mouseClickEvent)

    def toggle_drawing_mode(self) -> None:
        self.drawing = not self.drawing
        if self.drawing:
            self.scatter_plot.setMouseEnabled(x=False, y=False)
            self.scatter_plot.setMenuEnabled(False)
        else:
            self.scatter_plot.setMouseEnabled(x=True, y=True)
            self.scatter_plot.setMenuEnabled(True)

    def mouseClickEvent(self, event: Any) -> None:
        if self.drawing:
            if event.button() == Qt.LeftButton:
                logger.info("Left click detected")

            elif event.button() == Qt.RightButton:
                logger.info("Right click detected")

    # def mousePressEvent(self, event):
    #     if event.button() == QtCore.Qt.LeftButton:
    #         # Left mouse button: start or continue drawing the lasso
    #         #if not self.lassoPoints:
    #         self.lassoPoints = [self.vb.mapSceneToView(event.scenePos())]
    #         self.lasso = QtGui.QPolygonF(self.lassoPoints)
    #         self.lassoItem.setPolygon(self.lasso)

    #         self.setMouseEnabled(x=False, y=False)  # Disable default dragging behavior
    #         event.accept()
    #     elif event.button() == QtCore.Qt.RightButton:

    #         # Right mouse click

    #         self.lassoPoints = []
    #         self.lasso.clear()
    #         self.lassoItem.setPolygon(self.lasso)
    #         self.setMouseEnabled(x=True, y=True) # Re-enable default dragging behavior
    #         self.resetPointsColor()
    #         event.accept()
    #     else:
    #         super().mousePressEvent(event)

    # def mouseMoveEvent(self, event):
    #     if self.lassoPoints:
    #         self.lassoPoints.append(self.vb.mapSceneToView(event.scenePos()))
    #         self.lasso = QtGui.QPolygonF(self.lassoPoints)
    #         self.lassoItem.setPolygon(self.lasso)
    #         event.accept()
    #     else:
    #         super().mouseMoveEvent(event)

    # def mouseReleaseEvent(self, event):
    #     if self.lassoPoints and event.button() == QtCore.Qt.LeftButton:

    #         self.highlightPointsWithinLasso()
    #         self.setMouseEnabled(x=True, y=True) # Re-enable default dragging behavior

    #         event.accept()
    #     else:
    #         super().mouseReleaseEvent(event)

    # def highlightPointsWithinLasso(self):
    #     # Change color of points inside the lasso to red
    #     for i, point in enumerate(self.scatter.points()):
    #         if self.lasso.containsPoint(point.pos(), QtCore.Qt.OddEvenFill):
    #             point.setBrush(pg.mkBrush('r'))

    # def resetPointsColor(self):
    #     # Reset points to their original color
    #     for i, point in enumerate(self.scatter.points()):
    #         point.setBrush(self.org_brush[i])

    def use_auto_range(self) -> None:
        """Uses the auto-ranging feature for the plot"""
        self.scatter_plot.enableAutoRange("xy", True)

    def _onClick(
        self,
        x_data: NDArrayA | pd.Series,
        y_data: NDArrayA | pd.Series,
        color_data: NDArrayA | dict[str, NDArrayA | pd.Series | dict[str, str]],
        x_label: str | None,
        y_label: str | None,
        color_label: str | None,
    ) -> None:
        self.cat = None
        self.palette = None

        if isinstance(color_data, dict):
            self.data = [x_data, y_data, color_data["vec"]]
            self.cat = color_data["cat"]
            self.palette = color_data["palette"]

        else:
            norm = plt.colors.Normalize(vmin=np.amin(color_data), vmax=np.amax(color_data))
            cmap = plt.cm.viridis  # TODO (rahulbshrestha): Replace this with colormap used in scatterplot
            self.data = [x_data, y_data, cmap(norm(color_data))]

        self.x_label = x_label
        self.y_label = y_label
        self.color_label = color_label

        self.plot()

    def plot(self) -> None:

        logger.info("Generating scatter plot...")

        self.clear_plot()
        self.scatter_plot.setLabel("bottom", self.x_label)
        self.scatter_plot.setLabel("left", self.y_label)

        colors = [QColor(*[int(255 * c) for c in color]) for color in self.data[2]]
        brush = [pg.mkBrush(color) for color in colors]

        self.scatter = self.scatter_plot.plot(self.data[0], self.data[1], pen=None, symbol="o", symbolBrush=brush)

        self.scatter_plot.enableAutoRange("xy", True)

    def clear_plot(self) -> None:
        """Clears the scatter plot"""
        if self.scatter:
            self.scatter_plot.removeItem(self.scatter)
            self.scatter = None


class AxisWidgets(QtWidgets.QWidget):
    def __init__(self, model: DataModel, name: str, color: bool = False):
        super().__init__()

        self._model = model

        selection_label = QtWidgets.QLabel(f"{name} type:")
        selection_label.setToolTip("Select between obs, obsm and var.")
        self.selection_widget = QtWidgets.QComboBox()
        self.selection_widget.addItem("obsm", None)
        self.selection_widget.addItem("obs", None)
        self.selection_widget.addItem("var", None)

        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().addWidget(selection_label)
        self.layout().addWidget(self.selection_widget)

        label = QtWidgets.QLabel(f"Select for {name}:")
        label.setToolTip(f"Select {name}.")

        self.widget = ScatterListWidget(self.model, attr="obsm", color=color)
        self.widget.setAttribute("obsm")

        self.component_widget = ComponentWidget(self.model, attr="obsm")
        self.component_widget.setToolTip("obsm")
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
