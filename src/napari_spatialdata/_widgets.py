from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any

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
from spatialdata._types import ArrayLike

from napari_spatialdata._model import DataModel
from napari_spatialdata.utils._utils import _min_max_norm, get_napari_version

__all__ = ["AListWidget", "ComponentWidget"]

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
        if attr != "None" and attr not in DataModel.VALID_ATTRIBUTES:
            raise ValueError(f"Invalid attribute `{attr}`. Valid options are `{sorted(DataModel.VALID_ATTRIBUTES)}`.")
        super().__init__(viewer, **kwargs)

        self._viewer = viewer
        self._model = model

        self._attr = attr

        if attr == "None":
            self._getter: Callable[..., Any] = lambda: None
        else:
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
                vec, name, index = self._getter(item, index=self.getIndex())

                if self.model.layer is not None:
                    # update the features (properties for each instance displayed on mouse hover in the bottom bar)
                    self.getIndex()
                    features_name = f"{item}_{self.getIndex()}" if self._attr == "obsm" else item
                    features = pd.DataFrame({features_name: vec})
                    # we need this secret column "index", as explained here
                    # https://forum.image.sc/t/napari-labels-layer-properties/57649/2
                    features["index"] = index
                    self.model.layer.features = features

                    properties = self._get_points_properties(vec, key=item, layer=self.model.layer)
                    self.model.color_by = "" if self.model.system_name is None else item
                    if isinstance(self.model.layer, Points | Shapes):
                        self.model.layer.text = None  # needed because of the text-feature order of updates
                        self.model.layer.face_color = properties["face_color"]
                        # self.model.layer.edge_color = properties["face_color"]
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
    def _get_points_properties(self, vec: ArrayLike | pd.Series, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError(type(vec))

    @_get_points_properties.register(pd.Series)
    def _(self, vec: pd.Series, **kwargs: Any) -> dict[str, Any]:
        layer = kwargs.pop("layer", None)
        layer_metadata = self.model.layer.metadata if self.model.layer is not None else None
        if layer_metadata is None:
            raise ValueError("Layer metadata is not available.")
        element_indices = pd.Series(layer_metadata["indices"], name="element_indices")
        if isinstance(layer, Labels):
            element_indices = element_indices[element_indices != 0]
        # When merging if the row is not present in the other table it will be nan so we can give it a default color
        vec_color_name = vec.name + "_colors"
        if self._attr != "columns_df":
            if vec_color_name not in self.model.adata.uns:
                colorer = AnnData(shape=(len(vec), 0), obs=pd.DataFrame(index=vec.index, data={"vec": vec}))
                _set_colors_for_categorical_obs(colorer, "vec", palette="tab20")
                colors = colorer.uns["vec_colors"]
                color_dict = dict(zip(vec.cat.categories, colors, strict=False))
                color_dict.update({np.nan: "#808080ff"})
            else:
                colors = self.model.adata.uns[vec_color_name]
                color_dict = dict(zip(vec.cat.categories, colors.tolist(), strict=True))
        else:
            df = layer.metadata["_columns_df"]
            if vec_color_name not in df.columns:
                colorer = AnnData(shape=(len(vec), 0), obs=pd.DataFrame(index=vec.index, data={"vec": vec}))
                _set_colors_for_categorical_obs(colorer, "vec", palette="tab20")
                colors = colorer.uns["vec_colors"]
                color_dict = dict(zip(vec.cat.categories, colors, strict=False))
                color_dict.update({np.nan: "#808080ff"})
                color_column = vec.apply(lambda x: color_dict[x])
                df[vec_color_name] = color_column
            else:
                unique_colors = df[[vec.name, vec_color_name]].drop_duplicates()
                unique_colors.set_index(vec.name, inplace=True)
                if not unique_colors.index.is_unique:
                    raise ValueError(
                        f"The {vec_color_name} column must have unique values for the each {vec.name} level. Found:\n"
                        f"{unique_colors}"
                    )
                color_dict = unique_colors.to_dict()[f"{vec.name}_colors"]

        if self.model.instance_key is not None and self.model.instance_key == vec.index.name:
            merge_df = pd.merge(
                element_indices, vec, left_on="element_indices", right_on=self.model.instance_key, how="left"
            )
        else:
            merge_df = pd.merge(element_indices, vec, left_on="element_indices", right_index=True, how="left")

        merge_df["color"] = merge_df[vec.name].map(color_dict)
        if layer is not None and isinstance(layer, Labels):
            index_color_mapping = dict(zip(merge_df["element_indices"], merge_df["color"], strict=False))
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
    def _(self, vec: ArrayLike, **kwargs: Any) -> dict[str, Any]:
        layer = kwargs.pop("layer", None)

        # Here kwargs['key'] is actually the column name.
        column_df = False
        if (
            (adata := self.model.adata) is not None
            and kwargs["key"] not in adata.obs.columns
            and kwargs["key"] not in adata.var.index
            and kwargs["key"] not in adata.obsm
        ) or adata is None:
            merge_vec = layer.metadata["_columns_df"][kwargs["key"]]
            element_indices = merge_vec.index
            column_df = True
        else:
            instance_key_col = self.model.adata.obs[self.model.instance_key]
            vec = pd.Series(vec, name="vec", index=instance_key_col)
            layer_metadata = self.model.layer.metadata if self.model.layer is not None else None
            if layer_metadata is None:
                raise ValueError("Layer metadata is not available.")
            element_indices = pd.Series(layer_metadata["indices"], name="element_indices")
            if isinstance(layer, Labels):
                vec = vec.drop(index=0) if 0 in vec.index else vec
            # element_indices = element_indices[element_indices != 0]
            diff_element_table = set(element_indices).difference(set(vec.index))
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

        if layer is not None and isinstance(layer, Labels):
            return {
                "color": dict(zip(element_indices, color_vec, strict=False)),
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
    def __init__(self, model: DataModel, attr: str | None, max_visible: int = 4, **kwargs: Any):
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
        if field == "None":
            field = None
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


class ScatterAnnotationDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

        self.setWindowTitle("Name Obs")

        self.layout = QtWidgets.QVBoxLayout()

        self.label = QtWidgets.QLabel("Annotation Name:")
        self.layout.addWidget(self.label)

        self.textbox = QtWidgets.QLineEdit(self)
        self.layout.addWidget(self.textbox)

        self.buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout.addWidget(self.buttonBox)

        self.setLayout(self.layout)

    def get_annotation_name(self) -> str:
        return str(self.textbox.text())


class AnnDataSaveDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)

    def show_dialog(self) -> str | None:

        # Define file filters
        file_filters = "All Files (*);;H5AD Files (*.h5ad);;Zarr Files (*.zarr);; Csv Files (*.csv)"

        # Open the file dialog with the specified options and filters
        filePath: str
        selected_filter: str
        filePath, selected_filter = QtWidgets.QFileDialog.getSaveFileName(self, "Save AnnData", "", file_filters)

        if filePath:
            # Add the correct extension if not provided
            if selected_filter == "H5AD Files (*.h5ad)" and not filePath.endswith(".h5ad"):
                filePath += ".h5ad"
            elif selected_filter == "Zarr Files (*.zarr)" and not filePath.endswith(".zarr"):
                filePath += ".zarr"
            elif selected_filter == "Text Files (*.csv)" and not filePath.endswith(".csv"):
                filePath += ".csv"

            return filePath

        return None
