from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import matplotlib.pyplot as plt
import napari
import numpy as np
import packaging.version
import pandas as pd
from anndata import AnnData
from loguru import logger
from napari.layers import Image, Labels, Layer, Points, Shapes
from napari.utils import DirectLabelColormap
from napari.viewer import Viewer
from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Qt, Signal
from scanpy.plotting._utils import _set_colors_for_categorical_obs
from sklearn.preprocessing import MinMaxScaler
from superqt import QRangeSlider
from vispy import scene
from vispy.color.colormap import Colormap, MatplotlibColormap
from vispy.scene.widgets import ColorBarWidget

from napari_spatialdata._model import DataModel
from napari_spatialdata.utils._utils import NDArrayA, _min_max_norm, get_napari_version

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

    def __init__(self, viewer: napari.Viewer | None, unique: bool = True, multiselect: bool = True, **kwargs: Any):
        super().__init__(**kwargs)
        if multiselect:
            self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        else:
            self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        self._index: int | str = 0
        self._unique = unique
        self._viewer = viewer

        self.itemDoubleClicked.connect(lambda item: self._onAction((item.text(),)))
        self.enterPressed.connect(self._onAction)
        self.indexChanged.connect(self._onAction)

    @abstractmethod
    def setIndex(self, index: int | str) -> None:
        pass

    def getIndex(self) -> int | str:
        return self._index

    @abstractmethod
    def _onAction(self, items: Iterable[str]) -> None:
        pass

    def addItems(self, labels: str | Iterable[str] | None) -> None:
        if labels is None:
            return
        if isinstance(labels, str):
            labels = (labels,)
        labels = tuple(labels)

        if self._unique:
            labels = tuple(label for label in labels if self.findItems(label, QtCore.Qt.MatchExactly) is not None)

        if len(labels):
            super().addItems(labels)
            # self.sortItems(QtCore.Qt.AscendingOrder)

    def keyPressEvent(self, event: QtCore.QEvent) -> None:
        if event.key() == QtCore.Qt.Key_Return:
            event.accept()
            self.enterPressed.emit(tuple(s.text() for s in self.selectedItems()))
        else:
            super().keyPressEvent(event)


class AListWidget(ListWidget):
    layerChanged = Signal()

    def __init__(self, viewer: Viewer | None, model: DataModel, attr: str, **kwargs: Any):
        if attr not in DataModel.VALID_ATTRIBUTES:
            raise ValueError(f"Invalid attribute `{attr}`. Valid options are `{sorted(DataModel.VALID_ATTRIBUTES)}`.")
        super().__init__(viewer, **kwargs)

        self._viewer = viewer
        self._model = model

        self._attr = attr

        self._getter = getattr(self.model, f"get_{attr}")
        self.layerChanged.connect(self._onChange)
        self._onChange()

    def _onChange(self) -> None:
        self.clear()
        if self._model.adata is not None:
            self.addItems(self.model.get_items(self._attr))

    def _onAction(self, items: Iterable[str]) -> None:
        for item in sorted(set(items)):
            if isinstance(self.model.layer, (Image)):
                i = self.model.layer.metadata["adata"].var.index.get_loc(item)
                self.viewer.dims.set_point(0, i)
            else:
                vec, name = self._getter(item, index=self.getIndex())

                if self.model.layer is not None:
                    properties = self._get_points_properties(vec, key=item, layer=self.model.layer)
                    self.model.color_by = "" if self.model.system_name is None else item
                    if isinstance(self.model.layer, (Points, Shapes)):
                        self.model.layer.text = None  # needed because of the text-feature order of updates
                        # self.model.layer.features = properties.get("features", None)
                        self.model.layer.face_color = properties["face_color"]
                        self.model.layer.text = properties["text"]
                    elif isinstance(self.model.layer, Labels):
                        version = get_napari_version()
                        if version < packaging.version.parse("0.4.20"):
                            self.model.layer.color = properties["color"]
                            self.model.layer.properties = properties.get("properties", None)
                        else:
                            ddict = defaultdict(lambda: np.zeros(4), properties["color"])
                            cmap = DirectLabelColormap(color_dict=ddict)
                            self.model.layer.colormap = cmap
                    else:
                        raise ValueError("TODO")
                    # TODO(michalk8): add contrasting fg/bg color once https://github.com/napari/napari/issues/2019 is
                    #  done
                    # TODO(giovp): make layer editable?
                    # self.viewer.layers[layer_name].editable = False

    def setAdataLayer(self, layer: str | None) -> None:
        if layer in ("default", "None", "X"):
            layer = None
        if layer == self.getAdataLayer():
            return
        self.model.adata_layer = layer
        self.layerChanged.emit()

    def getAdataLayer(self) -> str | None:
        if TYPE_CHECKING:
            assert isinstance(self.model.adata_layer, str)
        return self.model.adata_layer

    def setIndex(self, index: int | str) -> None:
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

    def getIndex(self) -> int | str:
        return self._index

    def _handle_already_present(self, layer_name: str) -> None:
        logger.debug(f"Layer `{layer_name}` is already loaded")
        self.viewer.layers.selection.select_only(self.viewer.layers[layer_name])

    @singledispatchmethod
    def _get_points_properties(self, vec: NDArrayA | pd.Series, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError(type(vec))

    @_get_points_properties.register(pd.Series)
    def _(self, vec: pd.Series, **kwargs: Any) -> dict[str, Any]:
        layer = kwargs.pop("layer", None)
        layer_meta = self.model.layer.metadata if self.model.layer is not None else None
        element_indices = pd.Series(layer_meta["indices"], name="element_indices")
        if isinstance(layer, Labels):
            element_indices = element_indices[element_indices != 0]
        # When merging if the row is not present in the other table it will be nan so we can give it a default color
        if (vec_color_name := vec.name + "_color") not in self.model.adata.uns:
            colorer = AnnData(shape=(len(vec), 0), obs=pd.DataFrame(index=vec.index, data={"vec": vec}))
            _set_colors_for_categorical_obs(colorer, "vec", palette="tab20")
            colors = colorer.uns["vec_colors"]
            color_dict = dict(zip(vec.cat.categories, colors))
            color_dict.update({np.nan: "#808080ff"})
        else:
            color_dict = self.model.adata.uns[vec_color_name]

        if self.model.instance_key is not None and self.model.instance_key == vec.index.name:
            merge_df = pd.merge(
                element_indices, vec, left_on="element_indices", right_on=self.model.instance_key, how="left"
            )
        else:
            merge_df = pd.merge(element_indices, vec, left_on="element_indices", right_index=True, how="left")

        merge_df["color"] = merge_df[vec.name].map(color_dict)
        if layer is not None and isinstance(layer, Labels):
            index_color_mapping = dict(zip(merge_df["element_indices"], merge_df["color"]))
            index_color_mapping[0] = "#000000ff"
            return {
                "color": index_color_mapping,
                "properties": {"value": vec},
                "text": None,
            }

        return {
            "text": None,
            "face_color": merge_df["color"].to_list(),
        }

    @_get_points_properties.register(np.ndarray)
    def _(self, vec: NDArrayA, **kwargs: Any) -> dict[str, Any]:
        layer = kwargs.pop("layer", None)

        # Here kwargs['key'] is actually the column name.
        column_df = False
        if (
            (adata := self.model.adata) is not None
            and kwargs["key"] not in adata.obs.columns
            and kwargs["key"] not in adata.var.index
        ) or adata is None:
            merge_vec = layer.metadata["_columns_df"][kwargs["key"]]
            element_indices = merge_vec.index
            column_df = True
        else:
            instance_key_col = self.model.adata.obs[self.model.instance_key]
            vec = pd.Series(vec, name="vec", index=instance_key_col)
            layer_meta = self.model.layer.metadata if self.model.layer is not None else None
            element_indices = pd.Series(layer_meta["indices"], name="element_indices")
            if isinstance(layer, Labels):
                vec = vec.drop(index=0) if 0 in vec.index else vec  # type:ignore[attr-defined]
            # element_indices = element_indices[element_indices != 0]
            diff_element_table = set(element_indices).difference(set(vec.index))  # type:ignore[attr-defined]
            merge_vec = pd.merge(element_indices, vec, left_on="element_indices", right_index=True, how="left")[
                "vec"
            ].fillna(0, axis=0)

        cmap = plt.get_cmap(self.model.cmap)
        norm_vec = _min_max_norm(merge_vec)
        color_vec = cmap(norm_vec)

        if not column_df:
            element_indices_list = None
            for i in diff_element_table:
                if element_indices_list is None:
                    element_indices_list = element_indices.to_list()
                change_index = element_indices_list.index(i)
                color_vec[change_index] = np.array([0.5, 0.5, 0.5, 1.0])
            if isinstance(layer, Labels):
                color_vec[0] = np.array([0.0, 0.0, 0.0, 1.0])

        if layer is not None and isinstance(layer, Labels):
            return {
                "color": dict(zip(element_indices, color_vec)),
                "properties": {"value": vec},
                "text": None,
            }

        if layer is not None and isinstance(layer, Shapes):
            return {
                "text": None,
                "face_color": color_vec,
            }

        return {
            "text": None,
            "face_color": color_vec,
        }

    @property
    def viewer(self) -> napari.Viewer:
        """:mod:`napari` viewer."""
        return self._viewer

    @property
    def model(self) -> DataModel:
        """:mod:`napari` viewer."""
        return self._model


class ComponentWidget(QtWidgets.QComboBox):
    def __init__(self, model: DataModel, attr: str, max_visible: int = 4, **kwargs: Any):
        super().__init__(**kwargs)

        self._model = model
        self.view().setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setMaxVisibleItems(max_visible)
        self.setStyleSheet("combobox-popup: 0;")
        self._attr = attr

    def addItems(self, texts: QtWidgets.QListWidgetItem | int | Iterable[str]) -> None:
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

    def setAttribute(self, field: str | None) -> None:
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

    def _onClickChange(self, clicked: QtWidgets.QListWidgetItem | int | Iterable[str]) -> None:
        if self.attr == "obsm":
            self.clear()
            self.addItems(clicked)

    def _getAllLayers(self) -> Sequence[str | None]:
        adata_layers = list(self._model.adata.layers.keys())
        if len(adata_layers):
            adata_layers.insert(0, "X")
            return adata_layers
        return ["X"]

    @property
    def attr(self) -> str | None:
        if TYPE_CHECKING:
            assert isinstance(self._attr, str)
        return self._attr

    @attr.setter
    def attr(self, field: str | None) -> None:
        if field not in ("var", "obs", "obsm"):
            raise ValueError(f"{field} is not a valid adata field.")
        self._attr = field


class CBarWidget(QtWidgets.QWidget):
    FORMAT = "{0:0.2f}"

    cmapChanged = Signal(str)
    climChanged = Signal((float, float))

    def __init__(
        self,
        model: DataModel,
        cmap: str = "viridis",
        label: str | None = None,
        width: int | None = 250,
        height: int | None = 50,
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
    def __init__(self, viewer: Viewer, model: DataModel, colorbar: CBarWidget, **kwargs: Any):
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
    def model(self) -> DataModel:
        """:mod:`napari` viewer."""
        return self._model


class SaveDialog(QtWidgets.QDialog):
    def __init__(self, layer: Layer, table_name: str) -> None:
        super().__init__()

        self.setWindowTitle("Save Dialog")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.table_name: str | None = table_name if table_name != "" else f"annotation_{layer.name}"
        self.shape_name: str | None = layer.name
        self.sdata = layer.metadata["sdata"]

        layout = QtWidgets.QVBoxLayout(self)

        spatial_element_label = QtWidgets.QLabel("Spatial Element name:")
        self.spatial_element_line_edit = QtWidgets.QLineEdit(layer.name)
        layout.addWidget(spatial_element_label)
        layout.addWidget(self.spatial_element_line_edit)

        if any("color" in col for col in layer.features.columns):
            table_label = QtWidgets.QLabel("Table name:")
            self.table_line_edit = QtWidgets.QLineEdit(self.table_name)
            layout.addWidget(table_label)
            layout.addWidget(self.table_line_edit)

        QBtn = QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        self.button_box = QtWidgets.QDialogButtonBox(QBtn)
        self.button_box.accepted.connect(self.save_clicked)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def save_clicked(self) -> None:
        self.table_name = self.table_line_edit.text()
        self.shape_name = self.spatial_element_line_edit.text()
        if (overwrite_table := self.table_name in self.sdata.tables) or (
            overwrite_shape := self.shape_name in self.sdata.shapes
        ):
            overwrite_shape = self.shape_name in self.sdata.shapes

            if overwrite_table and overwrite_shape:
                reply = QtWidgets.QMessageBox.question(
                    self,
                    "Overwrite",
                    f"{self.shape_name} and {self.table_name} already exist. Do you want to " f"overwrite them?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No,
                )
            elif overwrite_shape:
                reply = QtWidgets.QMessageBox.question(
                    self,
                    "Overwrite",
                    f"{self.shape_name}  already exists. Do you want to overwrite?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No,
                )
            elif overwrite_table:
                reply = QtWidgets.QMessageBox.question(
                    self,
                    "Overwrite",
                    f"{self.table_name}  already exists. Do you want to overwrite?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No,
                )
            if reply == QtWidgets.QMessageBox.No:
                self.reject()
                return

        self.accept()

    def reject(self) -> None:
        self.table_name = None
        self.shape_name = None
        self.done(QtWidgets.QDialog.Rejected)

    def get_save_table_name(self) -> str | None:
        return getattr(self, "table_name", None)

    def get_save_shape_name(self) -> str | None:
        return getattr(self, "shape_name", None)
