from __future__ import annotations

from abc import abstractmethod
from typing import Any, Union, Iterable, Optional, TYPE_CHECKING
from functools import singledispatchmethod

from qtpy import QtCore, QtWidgets
from vispy import scene
from scanpy import logging as logg
from superqt import QRangeSlider
from qtpy.QtCore import Qt, Signal
from napari.layers import Image, Layer, Labels, Points
from qtpy.QtWidgets import QLabel, QWidget, QGridLayout
from napari.utils.events import Event, EmitterGroup
from vispy.scene.widgets import ColorBarWidget
from vispy.color.colormap import Colormap, MatplotlibColormap
import numpy as np
import napari
import pandas as pd
import matplotlib.pyplot as plt

from napari_spatialdata._model import ImageModel
from napari_spatialdata._utils import ALayer, NDArrayA, _min_max_norm, _get_categorical

__all__ = ["TwoStateCheckBox", "AListWidget", "CBarWidget", "RangeSlider", "ObsmIndexWidget", "LibraryListWidget"]

# label string: attribute name
_WIDGETS_TO_HIDE = {
    "symbol:": "symbolComboBox",
    "point size:": "sizeSlider",
    "face color:": "faceColorEdit",
    "edge color:": "edgeColorEdit",
    "out of slice:": "outOfSliceCheckBox",
}


class ListWidget(QtWidgets.QListWidget):
    indexChanged = Signal(object)
    enterPressed = Signal(object)

    def __init__(self, viewer: napari.Viewer, unique: bool = True, multiselect: bool = True, **kwargs: Any):
        super().__init__(**kwargs)
        if multiselect:
            self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        else:
            self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        self._index: Union[int, str] = 0
        self._unique = unique
        self._viewer = viewer

        self.itemDoubleClicked.connect(lambda item: self._onAction((item.text(),)))
        self.enterPressed.connect(self._onAction)
        self.indexChanged.connect(self._onAction)

    @abstractmethod
    def setIndex(self, index: Union[int, str]) -> None:
        pass

    def getIndex(self) -> Union[int, str]:
        return self._index

    @abstractmethod
    def _onAction(self, items: Iterable[str]) -> None:
        pass

    def addItems(self, labels: Union[str, Iterable[str]]) -> None:
        if isinstance(labels, str):
            labels = (labels,)
        labels = tuple(labels)

        if self._unique:
            labels = tuple(label for label in labels if self.findItems(label, QtCore.Qt.MatchExactly) is not None)

        if len(labels):
            super().addItems(labels)
            self.sortItems(QtCore.Qt.AscendingOrder)

    def keyPressEvent(self, event: QtCore.QEvent) -> None:
        if event.key() == QtCore.Qt.Key_Return:
            event.accept()
            self.enterPressed.emit(tuple(s.text() for s in self.selectedItems()))
        else:
            super().keyPressEvent(event)


class LibraryListWidget(ListWidget):
    def __init__(self, controller: Any, **kwargs: Any):
        super().__init__(controller, **kwargs)

        self.currentTextChanged.connect(self._onAction)

    def setIndex(self, index: Union[int, str]) -> None:
        # not used
        if index == self._index:
            return

        self._index = index
        self.indexChanged.emit(tuple(s.text() for s in self.selectedItems()))

    def _onAction(self, items: Union[str, Iterable[str]]) -> None:
        if isinstance(items, str):
            items = (items,)

        for item in items:
            if self._controller.add_image(item):
                # only add 1 item
                break


class TwoStateCheckBox(QtWidgets.QCheckBox):
    checkChanged = Signal(bool)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.setTristate(False)
        self.setChecked(False)
        self.stateChanged.connect(self._onStateChanged)

    def _onStateChanged(self, state: QtCore.Qt.CheckState) -> None:
        self.checkChanged.emit(state == QtCore.Qt.Checked)


class AListWidget(ListWidget):
    layerChanged = Signal()
    libraryChanged = Signal()

    def __init__(self, viewer: napari.Viewer, model: ImageModel, attr: str, **kwargs: Any):
        if attr not in ALayer.VALID_ATTRIBUTES:
            raise ValueError(f"Invalid attribute `{attr}`. Valid options are `{sorted(ALayer.VALID_ATTRIBUTES)}`.")
        super().__init__(viewer, **kwargs)

        self.viewer = viewer
        self.model = model
        self.events = EmitterGroup(
            source=self,
            layer=Event,
        )

        self._attr = attr
        self._getter = getattr(self.model, f"get_{attr}")

        self.layerChanged.connect(self._onChange)
        self.libraryChanged.connect(self._onChange)

        self._onChange()

    def _onChange(self) -> None:
        self.clear()
        self.addItems(self.model.get_items(self._attr))

    def _onAction(self, items: Iterable[str]) -> None:
        for item in sorted(set(items)):
            try:
                vec, name = self._getter(item, index=self.getIndex())
            except Exception as e:  # noqa: B902
                logg.error(e)
                continue
            layer_name = name
            properties = self._get_points_properties(vec, key=item, layer=self.model.layer)
            if isinstance(self.model.layer, Image):
                self.viewer.add_points(
                    self.model.coordinates,
                    name=layer_name,
                    size=self.model.spot_diameter,
                    opacity=1,
                    blending=self.model.blending,
                    face_colormap=self.model.cmap,
                    edge_colormap=self.model.cmap,
                    symbol=self.model.symbol,
                    **properties,
                )
            elif isinstance(self.model.layer, Labels):
                self.viewer.add_labels(
                    self.model.layer.data.copy(),
                    name=layer_name,
                    **properties,
                )
            else:
                raise ValueError("TODO")
            # TODO(michalk8): add contrasting fg/bg color once https://github.com/napari/napari/issues/2019 is done
            # TODO(giovp): grid_layout not working?
            # self._hide_points_controls(self.viewer.layers[layer_name], is_categorical=is_categorical_dtype(vec))
            # self.viewer.layers[layer_name].editable = False

    def setAdataLayer(self, layer: Optional[str]) -> None:
        if layer in ("default", "None"):
            layer = None
        if layer == self.getAdataLayer():
            return
        self.model.adata_layer = layer
        self.layerChanged.emit()

    def getAdataLayer(self) -> Optional[str]:
        return self.model.adata_layer

    def setIndex(self, index: Union[int, str]) -> None:
        if isinstance(index, str):
            if index == "":
                index = 0
            elif self._attr != "obsm":
                index = int(index, base=10)
            # for obsm, we convert index to int if needed (if not a DataFrame) in the ALayer
        if index == self._index:
            return

        self._index = index
        if self._attr == "obsm":
            self.indexChanged.emit(tuple(s.text() for s in self.selectedItems()))

    def getIndex(self) -> Union[int, str]:
        return self._index

    def _handle_already_present(self, layer_name: str) -> None:
        logg.debug(f"Layer `{layer_name}` is already loaded")
        self.viewer.layers.selection.select_only(self.viewer.layers[layer_name])

    @singledispatchmethod
    def _get_points_properties(self, vec: Union[NDArrayA, pd.Series], **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError(type(vec))

    @_get_points_properties.register(np.ndarray)
    def _(self, vec: NDArrayA, **kwargs: Any) -> dict[str, Any]:
        layer = kwargs.pop("layer", None)
        if layer is not None and isinstance(layer, Labels):
            cmap = plt.get_cmap(self.model.cmap)
            norm_vec = _min_max_norm(vec)
            color_vec = cmap(norm_vec)
            return {"color": {k: v for k, v in zip(self.model.adata.obs[self.model.labels_key].values, color_vec)}}
        return {
            "text": None,
            "face_color": "value",
            "properties": {"value": vec},
            "metadata": {"perc": (0, 100), "data": vec, "minmax": (np.nanmin(vec), np.nanmax(vec))},
        }

    @_get_points_properties.register(pd.Series)
    def _(self, vec: pd.Series, key: str, layer: Layer) -> dict[str, Any]:
        face_color = _get_categorical(self.model.adata, key=key, palette=self.model.palette, vec=vec)
        if layer is not None and isinstance(layer, Labels):
            return {"color": {k: v for k, v in zip(self.model.adata.obs[self.model.labels_key].values, face_color)}}
        return {
            "text": {"text": "{clusters}", "size": 24, "color": "white", "anchor": "center"},
            "face_color": face_color,
            # "properties": _position_cluster_labels(self.model.coordinates, vec, face_color),
            "metadata": None,
        }

    def _hide_points_controls(self, layer: Points, is_categorical: bool) -> None:
        try:
            # TODO(michalk8): find a better way: https://github.com/napari/napari/issues/3066
            points_controls = self.viewer.window._qt_viewer.controls.widgets[layer]
        except KeyError:
            return

        gl: QGridLayout = points_controls.grid_layout

        labels = {}
        for i in range(gl.count()):
            item = gl.itemAt(i).widget()
            if isinstance(item, QLabel):
                labels[item.text()] = item

        label_key, widget = "", None
        # remove all widgets which can modify the layer
        for label_key, widget_name in _WIDGETS_TO_HIDE.items():
            widget = getattr(points_controls, widget_name, None)
            if label_key in labels and widget is not None:
                widget.setHidden(True)
                labels[label_key].setHidden(True)

        if TYPE_CHECKING:
            assert isinstance(widget, QWidget)

        if not is_categorical:  # add the slider
            if widget is None:
                logg.warning("Unable to set the percentile slider")
                return
            idx = gl.indexOf(widget)
            row, *_ = gl.getItemPosition(idx)

            slider = RangeSlider(
                layer=layer,
                colorbar="viridis",  # TODO(giovp): add it to point layer widget
            )
            slider.valueChanged.emit((0, 100))
            gl.replaceWidget(labels[label_key], QLabel("percentile:"))
            gl.replaceWidget(widget, slider)


class ObsmIndexWidget(QtWidgets.QComboBox):
    def __init__(self, model: ImageModel, max_visible: int = 6, **kwargs: Any):
        super().__init__(**kwargs)

        self._model = model
        self.view().setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setMaxVisibleItems(max_visible)
        self.setStyleSheet("combobox-popup: 0;")

    def addItems(self, texts: Union[QtWidgets.QListWidgetItem, int, Iterable[str]]) -> None:
        if isinstance(texts, QtWidgets.QListWidgetItem):
            try:
                key = texts.text()
                if isinstance(self._model.adata.obsm[key], pd.DataFrame):
                    texts = sorted(self._model.adata.obsm[key].select_dtypes(include=[np.number, "category"]).columns)
                elif hasattr(self._model.adata.obsm[key], "shape"):
                    texts = self._model.adata.obsm[key].shape[1]
                else:
                    texts = np.asarray(self._model.adata.obsm[key]).shape[1]
            except (KeyError, IndexError):
                texts = 0
        if isinstance(texts, int):
            texts = tuple(str(i) for i in range(texts))

        self.clear()
        super().addItems(tuple(texts))


class CBarWidget(QtWidgets.QWidget):
    FORMAT = "{0:0.2f}"

    cmapChanged = Signal(str)
    climChanged = Signal((float, float))

    def __init__(
        self,
        cmap: Union[str, Colormap],
        label: Optional[str] = None,
        width: Optional[int] = 250,
        height: Optional[int] = 50,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self._cmap = cmap
        self._clim = (0.0, 1.0)
        self._oclim = self._clim

        self._width = width
        self._height = height
        self._label = label

        self.__init_UI()

    def __init_UI(self) -> None:
        self.setFixedWidth(self._width)
        self.setFixedHeight(self._height)

        # use napari's BG color for dark mode
        self._canvas = scene.SceneCanvas(
            size=(self._width, self._height), bgcolor="#262930", parent=self, decorate=False, resizable=False, dpi=150
        )
        self._colorbar = ColorBarWidget(
            self._create_colormap(self.getCmap()),
            orientation="top",
            label=self._label,
            label_color="white",
            clim=self.getClim(),
            border_width=1.0,
            border_color="black",
            padding=(0.33, 0.167),
            axis_ratio=0.05,
        )

        self._canvas.central_widget.add_widget(self._colorbar)

        self.climChanged.connect(self.onClimChanged)
        self.cmapChanged.connect(self.onCmapChanged)

    def _create_colormap(self, cmap: str) -> Colormap:
        ominn, omaxx = self.getOclim()
        delta = omaxx - ominn + 1e-12

        minn, maxx = self.getClim()
        minn = (minn - ominn) / delta
        maxx = (maxx - ominn) / delta

        assert 0 <= minn <= 1, f"Expected `min` to be in `[0, 1]`, found `{minn}`"
        assert 0 <= maxx <= 1, f"Expected `maxx` to be in `[0, 1]`, found `{maxx}`"

        cm = MatplotlibColormap(cmap)

        return Colormap(cm[np.linspace(minn, maxx, len(cm.colors))], interpolation="linear")

    def setCmap(self, cmap: str) -> None:
        if self._cmap == cmap:
            return

        self._cmap = cmap
        self.cmapChanged.emit(cmap)

    def getCmap(self) -> str:
        return self._cmap

    def onCmapChanged(self, value: str) -> None:
        # this does not trigger update for some reason...
        self._colorbar.cmap = self._create_colormap(value)
        self._colorbar._colorbar._update()

    def setClim(self, value: tuple[float, float]) -> None:
        if value == self._clim:
            return

        self._clim = value
        self.climChanged.emit(*value)

    def getClim(self) -> tuple[float, float]:
        return self._clim

    def getOclim(self) -> tuple[float, float]:
        return self._oclim

    def setOclim(self, value: tuple[float, float]) -> None:
        # original color limit used for 0-1 normalization
        self._oclim = value

    def onClimChanged(self, minn: float, maxx: float) -> None:
        # ticks are not working with vispy's colorbar
        self._colorbar.cmap = self._create_colormap(self.getCmap())
        self._colorbar.clim = (self.FORMAT.format(minn), self.FORMAT.format(maxx))

    def getCanvas(self) -> scene.SceneCanvas:
        return self._canvas

    def getColorBar(self) -> ColorBarWidget:
        return self._colorbar

    def setLayout(self, layout: QtWidgets.QLayout) -> None:
        layout.addWidget(self.getCanvas().native)
        super().setLayout(layout)

    def update_color(self) -> None:
        # when changing selected layers that have the same limit
        # could also trigger it as self._colorbar.clim = self.getClim()
        # but the above option also updates geometry
        # cbarwidget->cbar->cbarvisual
        self._colorbar._colorbar._colorbar._update()


class RangeSlider(QRangeSlider):
    def __init__(self, *args: Any, layer: Points, colorbar: CBarWidget, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self._layer = layer
        self._colorbar = colorbar
        self.setValue((0, 100))
        self.setSliderPosition((0, 100))
        self.setSingleStep(0.01)
        self.setOrientation(Qt.Horizontal)

        self.valueChanged.connect(self._onValueChange)

    def _onValueChange(self, percentile: tuple[float, float]) -> None:
        # TODO(michalk8): use constants
        v = self._layer.metadata["data"]
        clipped = np.clip(v, *np.percentile(v, percentile))

        self._layer.metadata = {**self._layer.metadata, "perc": percentile}
        self._layer.face_color = "value"
        self._layer.properties = {"value": clipped}
        self._layer._update_thumbnail()  # can't find another way to force it
        self._layer.refresh_colors()

        self._colorbar.setOclim(self._layer.metadata["minmax"])
        self._colorbar.setClim((np.min(self._layer.properties["value"]), np.max(self._layer.properties["value"])))
        self._colorbar.update_color()
