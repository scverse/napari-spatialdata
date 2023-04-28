from __future__ import annotations

from abc import abstractmethod
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence, Union

import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
from loguru import logger
from napari.layers import Image, Labels, Layer, Points
from napari.viewer import Viewer
from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Qt, Signal
from sklearn.preprocessing import MinMaxScaler
from superqt import QRangeSlider
from vispy import scene
from vispy.color.colormap import Colormap, MatplotlibColormap
from vispy.scene.widgets import ColorBarWidget

from napari_spatialdata._model import ImageModel
from napari_spatialdata._utils import (
    NDArrayA,
    _get_categorical,
    _min_max_norm,
    _position_cluster_labels,
    _set_palette,
)

__all__ = [
    "AListWidget",
    "CBarWidget",
    "RangeSliderWidget",
    "ComponentWidget",
]

# label string: attribute name
# TODO(giovp): remove since layer controls private?
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


class AListWidget(ListWidget):
    layerChanged = Signal()

    def __init__(self, viewer: Viewer, model: ImageModel, attr: str, **kwargs: Any):
        if attr not in ImageModel.VALID_ATTRIBUTES:
            raise ValueError(f"Invalid attribute `{attr}`. Valid options are `{sorted(ImageModel.VALID_ATTRIBUTES)}`.")
        super().__init__(viewer, **kwargs)

        self._viewer = viewer
        self._model = model

        self._attr = attr
        self._getter = getattr(self.model, f"get_{attr}")

        self.layerChanged.connect(self._onChange)
        self._onChange()

    def _onChange(self) -> None:
        self.clear()
        self.addItems(self.model.get_items(self._attr))

    def _onAction(self, items: Iterable[str]) -> None:
        for item in sorted(set(items)):
            try:
                vec, name = self._getter(item, index=self.getIndex())
            except Exception as e:  # noqa: BLE001
                logger.error(e)
                continue
            if vec.ndim == 2:
                self.viewer.add_points(
                    vec,
                    name=name,
                    edge_color="white",
                    face_color="white",
                    size=self.model.point_diameter,
                    symbol=self.model.symbol,
                )
            else:
                properties = self._get_points_properties(vec, key=item, layer=self.model.layer)
                if isinstance(self.model.layer, (Image, Points)):
                    self.viewer.add_points(
                        self.model.coordinates,
                        name=name,
                        size=self.model.spot_diameter,
                        opacity=1,
                        face_colormap=self.model.cmap,
                        edge_colormap=self.model.cmap,
                        symbol=self.model.symbol,
                        **properties,
                    )
                elif isinstance(self.model.layer, Labels):
                    self.viewer.add_labels(
                        self.model.layer.data.copy(),
                        name=name,
                        **properties,
                    )
                else:
                    raise ValueError("TODO")
                # TODO(michalk8): add contrasting fg/bg color once https://github.com/napari/napari/issues/2019 is done
                # TODO(giovp): make layer editable?
                # self.viewer.layers[layer_name].editable = False

    def setAdataLayer(self, layer: Optional[str]) -> None:
        if layer in ("default", "None", "X"):
            layer = None
        if layer == self.getAdataLayer():
            return
        self.model.adata_layer = layer
        self.layerChanged.emit()

    def getAdataLayer(self) -> Optional[str]:
        if TYPE_CHECKING:
            assert isinstance(self.model.adata_layer, str)
        return self.model.adata_layer

    def setIndex(self, index: Union[int, str]) -> None:
        if isinstance(index, str):
            if index == "":
                index = 0
            elif self._attr != "obsm":
                index = int(index, base=10)
            # for obsm, we convert index to int if needed (if not a DataFrame)
        if index == self._index:
            return

        self._index = index
        if self._attr == "obsm":
            self.indexChanged.emit(tuple(s.text() for s in self.selectedItems()))

    def getIndex(self) -> Union[int, str]:
        return self._index

    def _handle_already_present(self, layer_name: str) -> None:
        logger.debug(f"Layer `{layer_name}` is already loaded")
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
            return {
                "color": dict(zip(self.model.adata.obs[self.model.labels_key].values, color_vec)),
                "properties": {"value": vec},
                "metadata": {"perc": (0, 100), "data": vec, "minmax": (np.nanmin(vec), np.nanmax(vec))},
            }
        return {
            "text": None,
            "face_color": "value",
            "properties": {"value": vec},
            "metadata": {"perc": (0, 100), "data": vec, "minmax": (np.nanmin(vec), np.nanmax(vec))},
        }

    @_get_points_properties.register(pd.Series)
    def _(self, vec: pd.Series, key: str, layer: Layer) -> dict[str, Any]:
        colortypes = _set_palette(self.model.adata, key=key, palette=self.model.palette, vec=vec)
        face_color = _get_categorical(
            self.model.adata, key=key, palette=self.model.palette, colordict=colortypes, vec=vec
        )
        if layer is not None and isinstance(layer, Labels):
            return {"color": dict(zip(self.model.adata.obs[self.model.labels_key].values, face_color))}

        cluster_labels = _position_cluster_labels(self.model.coordinates, vec)
        return {
            "text": {
                "string": "{clusters}",
                "size": 24,
                "color": {"feature": "clusters", "colormap": colortypes},
                "anchor": "center",
            },
            "face_color": face_color,
            "features": cluster_labels,
            "metadata": None,
        }

    @property
    def viewer(self) -> napari.Viewer:
        """:mod:`napari` viewer."""
        return self._viewer

    @property
    def model(self) -> ImageModel:
        """:mod:`napari` viewer."""
        return self._model


class ComponentWidget(QtWidgets.QComboBox):
    def __init__(self, model: ImageModel, attr: str, max_visible: int = 4, **kwargs: Any):
        super().__init__(**kwargs)

        self._model = model
        self.view().setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setMaxVisibleItems(max_visible)
        self.setStyleSheet("combobox-popup: 0;")
        self._attr = attr

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

    def setToolTip(self, click: str) -> None:
        if click == "obsm":
            super().setToolTip("Indices for current key in `adata.obsm`. Choose by clicking on item from obsm list.")
        elif click == "var":
            super().setToolTip("Keys in `adata.layers`.")
        else:
            super().setToolTip("")
        return

    def setAttribute(self, field: Optional[str]) -> None:
        if field == self.attr:
            return
        self.attr = field
        self._onChange()

    def _onChange(self) -> None:
        if self.attr == "var":
            self.clear()
            super().addItems(self._getAllLayers())
        else:
            self.clear()

    def _onClickChange(self, clicked: Union[QtWidgets.QListWidgetItem, int, Iterable[str]]) -> None:
        if self.attr == "obsm":
            self.clear()
            self.addItems(clicked)

    def _getAllLayers(self) -> Sequence[Optional[str]]:
        adata_layers = list(self._model.adata.layers.keys())
        if len(adata_layers):
            adata_layers.insert(0, "X")
            return adata_layers
        return ["X"]

    @property
    def attr(self) -> Optional[str]:
        if TYPE_CHECKING:
            assert isinstance(self._attr, str)
        return self._attr

    @attr.setter
    def attr(self, field: Optional[str]) -> None:
        if field not in ("var", "obs", "obsm"):
            raise ValueError(f"{field} is not a valid adata field.")
        self._attr = field


class CBarWidget(QtWidgets.QWidget):
    FORMAT = "{0:0.2f}"

    cmapChanged = Signal(str)
    climChanged = Signal((float, float))

    def __init__(
        self,
        model: ImageModel,
        cmap: str = "viridis",
        label: Optional[str] = None,
        width: Optional[int] = 250,
        height: Optional[int] = 50,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self._model = model

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
            self._create_colormap(self.cmap),
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

    def getCmap(self) -> str:
        return self.cmap

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
        self._colorbar.cmap = self._create_colormap(self.cmap)
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

    @property
    def cmap(self) -> str:
        return self._model.cmap


class RangeSliderWidget(QRangeSlider):
    def __init__(self, viewer: Viewer, model: ImageModel, colorbar: CBarWidget, **kwargs: Any):
        super().__init__(**kwargs)

        self._viewer = viewer
        self._model = model
        self._colorbar = colorbar
        self._cmap = plt.get_cmap(self._colorbar.cmap)
        self.setValue((0, 100))
        self.setSliderPosition((0, 100))
        self.setSingleStep(0.01)
        self.setOrientation(Qt.Horizontal)
        self.valueChanged.connect(self._onValueChange)

    def _onLayerChange(self) -> None:
        layer = self.viewer.layers.selection.active
        if layer is not None:
            self._onValueChange((0, 100))

    def _onValueChange(self, percentile: tuple[float, float]) -> None:
        layer = self.viewer.layers.selection.active
        # TODO(michalk8): use constants
        if "data" not in layer.metadata:
            return None  # noqa: RET501
        v = layer.metadata["data"]
        clipped = np.clip(v, *np.percentile(v, percentile))

        if isinstance(layer, Points):
            layer.metadata = {**layer.metadata, "perc": percentile}
            layer.face_color = "value"
            layer.properties = {"value": clipped}
            layer.refresh_colors()
        elif isinstance(layer, Labels):
            norm_vec = self._scale_vec(clipped)
            color_vec = self._cmap(norm_vec)
            layer.color = dict(zip(layer.color.keys(), color_vec))
            layer.properties = {"value": clipped}
            layer.refresh()

        self._colorbar.setOclim(layer.metadata["minmax"])
        self._colorbar.setClim((np.min(layer.properties["value"]), np.max(layer.properties["value"])))
        self._colorbar.update_color()

    def _scale_vec(self, vec: NDArrayA) -> NDArrayA:
        ominn, omaxx = self._colorbar.getOclim()
        delta = omaxx - ominn + 1e-12

        minn, maxx = self._colorbar.getClim()
        minn = (minn - ominn) / delta
        maxx = (maxx - ominn) / delta
        scaler = MinMaxScaler(feature_range=(minn, maxx))
        return scaler.fit_transform(vec.reshape(-1, 1))  # type: ignore[no-any-return]

    @property
    def viewer(self) -> napari.Viewer:
        """:mod:`napari` viewer."""
        return self._viewer

    @property
    def model(self) -> ImageModel:
        """:mod:`napari` viewer."""
        return self._model
