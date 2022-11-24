from __future__ import annotations

from typing import Any, Union, Iterable, Optional, TYPE_CHECKING

from qtpy import QtWidgets
from loguru import logger
from qtpy.QtCore import Signal
from napari.viewer import Viewer
from napari_matplotlib.base import NapariMPLWidget
import numpy as np
import napari
import pandas as pd

from napari_spatialdata._model import ImageModel
from napari_spatialdata._utils import NDArrayA, _get_categorical
from napari_spatialdata._widgets import AListWidget, ComponentWidget

__all__ = [
    "MatplotlibWidget",
    "AxisWidgets",
]


class ScatterListWidget(AListWidget):
    attrChanged = Signal()
    _text = None
    _chosen = None

    def __init__(self, viewer: Viewer, model: ImageModel, attr: str, color: bool, **kwargs: Any):
        AListWidget.__init__(self, viewer, model, attr, **kwargs)
        self.attrChanged.connect(self._onChange)
        self._color = color
        self._data = None
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
            except Exception as e:  # noqa: B902
                logger.error(e)
                continue
            self.chosen = item
            if isinstance(vec, np.ndarray):
                self.data = vec
            elif vec.dtype == "category":
                self.data = vec
                if self._color:
                    self.data = _get_categorical(self.model.adata, key=item, palette=self.model.palette, vec=self.data)
            else:
                raise TypeError(f"The chosen field's datatype ({vec.dtype.name}) cannot be plotted")
        return

    def _onOneClick(self, items: Iterable[str]) -> None:
        if self.getAttribute() == "obsm":
            return
        self._onAction(items)
        return

    def setAttribute(self, field: Optional[str]) -> None:
        if field == self.getAttribute():
            return
        if field not in ("var", "obs", "obsm"):
            raise ValueError(f"{field} is not a valid adata field.")
        self._attr = field
        self._getter = getattr(self.model, f"get_{field}")
        self.attrChanged.emit()

    def getAttribute(self) -> Optional[str]:
        if TYPE_CHECKING:
            assert isinstance(self._attr, str)
        return self._attr

    def setComponent(self, text: Optional[Union[int, str]]) -> None:

        if self.getAttribute() == "var":
            if TYPE_CHECKING:
                assert isinstance(text, str)
            self.text = text
            super().setAdataLayer(text)
        elif self.getAttribute() == "obsm":
            if TYPE_CHECKING:
                assert isinstance(text, int) or isinstance(text, str)
            self.text = text  # type: ignore[assignment]
            super().setIndex(text)

    @property
    def text(self) -> Optional[str]:
        return self._text

    @text.setter
    def text(self, text: Optional[Union[str, int]]) -> None:
        self._text = text if text is not None else None

    @property
    def chosen(self) -> Optional[str]:
        return self._chosen

    @chosen.setter
    def chosen(self, chosen: Optional[str]) -> None:
        self._chosen = chosen if chosen is not None else None

    @property
    def data(self) -> Union[None, NDArrayA]:
        return self._data

    @data.setter
    def data(self, data: NDArrayA) -> None:
        self._data = data


class MatplotlibWidget(NapariMPLWidget):
    def __init__(self, viewer: Viewer, model: ImageModel):

        super().__init__(viewer)

        self._viewer = viewer
        self._model = model
        self.axes = self.canvas.figure.subplots()
        self.colorbar = None

    def _onClick(
        self,
        x_data: Union[NDArrayA, pd.Series],
        y_data: Union[NDArrayA, pd.Series],
        color_data: Union[NDArrayA, pd.Series],
        x_label: Optional[str],
        y_label: Optional[str],
        color_label: Optional[str],
    ) -> None:

        self.data = [x_data, y_data, color_data]
        self.x_label = x_label
        self.y_label = y_label
        self.color_label = color_label

        self.plot()

    def plot(self) -> None:

        logger.info("Plotting coordinates.")

        self.clear()

        self.scatterplot = self.axes.scatter(x=self.data[0], y=self.data[1], c=self.data[2])
        self.colorbar = self.canvas.figure.colorbar(self.scatterplot)
        self.axes.set_xlabel(self.x_label)
        self.axes.set_ylabel(self.y_label)

        if self.colorbar is None:
            raise ValueError("Colorbar hasn't been created.")

        self.colorbar.set_label(self.color_label)

        self.canvas.draw()

    def clear(self) -> None:

        self.axes.clear()

        if self.colorbar:
            self.colorbar.remove()


class AxisWidgets(QtWidgets.QWidget):
    def __init__(self, viewer: Viewer, model: ImageModel, name: str, color: bool = False):

        super().__init__()

        self._viewer = viewer
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

        self.widget = ScatterListWidget(self.viewer, self.model, attr="obsm", color=color)
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

    def getFormattedLabel(self) -> Optional[str]:

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

    @property
    def viewer(self) -> napari.Viewer:
        """:mod:`napari` viewer."""
        return self._viewer

    @property
    def model(self) -> ImageModel:
        """:mod:`napari` viewer."""
        return self._model
