from __future__ import annotations

from typing import Any, Union, Iterable, Optional, TYPE_CHECKING

from qtpy import QtWidgets
from loguru import logger
from qtpy.QtCore import Signal
from napari.viewer import Viewer
from napari_matplotlib.base import NapariMPLWidget
import numpy as np

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

    def __init__(self, viewer: Viewer, model: ImageModel, attr: str, color: bool, **kwargs: Any):
        AListWidget.__init__(self, viewer, model, attr, **kwargs)
        self.attrChanged.connect(self._onChange)
        self._color = color
        self._data = None

    def _onChange(self) -> None:
        AListWidget._onChange(self)
        self.data = None

    def _onAction(self, items: Iterable[str]) -> None:
        for item in sorted(set(items)):
            try:
                vec, _ = self._getter(item, index=self.getIndex())
            except Exception as e:  # noqa: B902
                logger.error(e)
                continue
            if isinstance(vec, np.ndarray):
                self.data = vec
            elif vec.dtype == "category":
                self.data = vec
                if self._color:
                    self.data = _get_categorical(self.model.adata, key=item, palette=self.model.palette, vec=self.data)
            else:
                raise TypeError(f"The chosen field's datatype ({vec.dtype.name}) cannot be plotted")
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
        self.text = text  # type: ignore[assignment]

        if self.getAttribute() == "var":
            if TYPE_CHECKING:
                assert isinstance(text, str)
            super().setAdataLayer(text)
        elif self.getAttribute() == "obsm":
            if TYPE_CHECKING:
                assert isinstance(text, int)
            super().setIndex(text)

    @property
    def text(self) -> Optional[str]:
        return self._text

    @text.setter
    def text(self, text: Optional[Union[str, int]]) -> None:
        self._text = str(text) if text is not None else None

    @property
    def data(self) -> Union[None, NDArrayA]:
        return self._data

    @data.setter
    def data(self, data: NDArrayA) -> None:
        self._data = data


class MatplotlibWidget(NapariMPLWidget):
    def __init__(self, viewer: Viewer, model: ImageModel):

        super().__init__(viewer)

        self.viewer = viewer
        self.model = model
        self.axes = self.canvas.figure.subplots()

    def _onClick(
        self,
        x_data: NDArrayA,
        x_label: Optional[str],
        y_data: NDArrayA,
        y_label: Optional[str],
        color_data: NDArrayA,
        color_label: Optional[str],
    ) -> None:

        logger.debug("X-axis Data: {}", x_data)  # noqa: P103
        logger.debug("X-axis Label: {}", x_label)  # noqa: P103
        logger.debug("Y-axis Data: {}", y_data)  # noqa: P103
        logger.debug("Y-axis Label: {}", y_label)  # noqa: P103
        logger.debug("Color Data: {}", color_data)  # noqa: P103
        logger.debug("Color Label: {}", color_label)  # noqa: P103

        self.clear()
        self.draw(x_data, x_label, y_data, y_label, color_data, color_label)

    def draw(
        self,
        x_data: NDArrayA,
        x_label: Optional[str],
        y_data: NDArrayA,
        y_label: Optional[str],
        color_data: NDArrayA,
        color_label: Optional[str],
    ) -> None:

        self.axes.scatter(x=x_data, y=y_data, c=color_data, alpha=0.5)
        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel(y_label)

    def clear(self) -> None:
        self.axes.clear()


class AxisWidgets(QtWidgets.QWidget):
    def __init__(self, viewer: Viewer, model: ImageModel, name: str, color: bool = False):
        super().__init__()

        self.viewer = viewer
        self.model = model
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
