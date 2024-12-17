from __future__ import annotations

import platform
from collections.abc import Iterable
from importlib.metadata import version
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import shapely
from napari.layers import Points, Shapes
from napari.utils.events import EventedList
from napari.utils.notifications import show_info
from packaging.version import parse as parse_version
from qtpy.QtCore import QThread, Signal
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QLabel, QListWidget, QListWidgetItem, QProgressBar, QVBoxLayout, QWidget
from spatialdata import SpatialData
from spatialdata.models._utils import DEFAULT_COORDINATE_SYSTEM

from napari_spatialdata._viewer import SpatialDataViewer
from napari_spatialdata.constants.config import N_CIRCLES_WARNING_THRESHOLD, N_SHAPES_WARNING_THRESHOLD
from napari_spatialdata.utils._utils import _get_sdata_key, get_duplicate_element_names, get_elements_meta_mapping

if TYPE_CHECKING:
    from napari import Viewer
    from napari.utils.events.event import Event

icon_path = Path(__file__).parent / "resources/exclamation.png"

# if run with numpy<2 on macOS arm64 architecture compiled from pypi wheels,
# then it will crash with bus error if numpy is used in different thread
# Issue reported https://github.com/numpy/numpy/issues/21799
if (
    parse_version(version("napari")) < parse_version("0.5.3")
    and parse_version(version("numpy")) < parse_version("2")
    and platform.system() == "Darwin"
    and platform.machine() == "arm64"
):  # pragma: no cover
    try:
        PROBLEMATIC_NUMPY_MACOS = "cibw-run" in np.show_config("dicts")["Python Information"]["path"]  # type: ignore[call-arg,func-returns-value,unused-ignore]
    except (KeyError, TypeError):
        PROBLEMATIC_NUMPY_MACOS = False
else:
    PROBLEMATIC_NUMPY_MACOS = False


class ElementWidget(QListWidget):
    def __init__(self, sdata: EventedList):
        super().__init__()
        self._icon = QIcon(str(icon_path))
        self._sdata = sdata
        self._duplicate_element_names, _ = get_duplicate_element_names(self._sdata)
        self._elements: None | dict[str, dict[str, str | int]] = None

    def _onItemChange(self, selected_coordinate_system: QListWidgetItem | int | Iterable[str]) -> None:
        self.clear()
        elements, _ = get_elements_meta_mapping(self._sdata, selected_coordinate_system, self._duplicate_element_names)
        self._set_element_widget_items(elements)
        self._elements = elements

    def _set_element_widget_items(self, elements: dict[str, dict[str, str | int]]) -> None:
        for key, dict_val in sorted(elements.items(), key=itemgetter(0)):
            sdata = self._sdata[dict_val["sdata_index"]]
            element_type = dict_val["element_type"]
            element_name = dict_val["original_name"]
            item = QListWidgetItem(key)
            if element_type == "shapes":
                if (
                    type((element := sdata.shapes[element_name]).iloc[0].geometry) is shapely.Point
                    and len(element) > N_CIRCLES_WARNING_THRESHOLD
                ):
                    item.setIcon(self._icon)
                    item.setToolTip(
                        "Visualizing this many circles is currently slow in napari. Consider whether you want to "
                        "visualize."
                    )
                    self.addItem(item)
                elif (
                    type((element := sdata.shapes[element_name]).iloc[0].geometry)
                    in [shapely.Polygon, shapely.MultiPolygon]
                    and len(element) > N_SHAPES_WARNING_THRESHOLD
                ):
                    item.setIcon(self._icon)
                    item.setToolTip(
                        "Visualizing this many shapes is currently slow in napari. Consider whether you want to "
                        "visualize."
                    )
            self.addItem(item)


class CoordinateSystemWidget(QListWidget):
    def __init__(self, sdata: EventedList):
        super().__init__()

        self._sdata = sdata
        self._system: None | str = None

        # Sort alphabetically, but keep default "global" at the top.
        coordinate_systems = sorted(cs for sdata in self._sdata for cs in sdata.coordinate_systems)
        if DEFAULT_COORDINATE_SYSTEM in coordinate_systems:
            coordinate_systems.remove(DEFAULT_COORDINATE_SYSTEM)
            coordinate_systems.insert(0, DEFAULT_COORDINATE_SYSTEM)
        self.addItems(coordinate_systems)

    def _select_coord_sys(self, selected_coordinate_system: QListWidgetItem | int | Iterable[str]) -> None:
        self._system = str(selected_coordinate_system)


class DataLoadThread(QThread):
    returned = Signal(object)

    def __init__(self, parent: SdataWidget):
        super().__init__(parent=parent)
        self.sdata_widget = parent
        self._data_type = ""
        self._text = ""
        self._sdata = None
        self._selected_cs: str = ""
        self._multi: bool = False

    def load_data(self, data_type: str, text: str, sdata: SpatialData, selected_cs: str, multi: bool) -> None:
        if self.isRunning():
            raise RuntimeError("Thread is already running.")
        self._data_type = data_type
        self._text = text
        self._sdata = sdata
        self._selected_cs = selected_cs
        self._multi = multi

        if PROBLEMATIC_NUMPY_MACOS:
            self.run()
        else:
            self.start()

    def run(self) -> None:
        if not self._data_type:
            return
        if self._data_type == "labels":
            layer = self.sdata_widget.viewer_model.get_sdata_labels(
                self._sdata, self._text, self._selected_cs, self._multi
            )
        elif self._data_type == "images":
            layer = self.sdata_widget.viewer_model.get_sdata_image(
                self._sdata, self._text, self._selected_cs, self._multi
            )
        elif self._data_type == "points":
            layer = self.sdata_widget.viewer_model.get_sdata_points(
                self._sdata, self._text, self._selected_cs, self._multi
            )
        elif self._data_type == "shapes":
            layer = self.sdata_widget._get_shapes(self._sdata, self._text, self._selected_cs, self._multi)
        else:
            raise ValueError(f"Data type {self._data_type} not recognized.")

        self.returned.emit(layer)


class SdataWidget(QWidget):
    def __init__(self, viewer: Viewer, sdata: EventedList):
        super().__init__()
        self._sdata = sdata
        self.viewer_model = SpatialDataViewer(viewer, self._sdata)
        self.worker_thread = DataLoadThread(parent=self)
        self.worker_thread.returned.connect(self.viewer_model.viewer.add_layer)
        self.worker_thread.finished.connect(self._hide_slider)

        self.setLayout(QVBoxLayout())

        self.coordinate_system_widget = CoordinateSystemWidget(self._sdata)
        self.elements_widget = ElementWidget(self._sdata)
        self.slider = QProgressBar(self)
        self.slider.setRange(0, 0)
        self.slider.setVisible(False)

        self.layout().addWidget(self.slider)
        self.layout().addWidget(QLabel("Coordinate System:"))
        self.layout().addWidget(self.coordinate_system_widget)
        self.layout().addWidget(QLabel("Elements:"))
        self.layout().addWidget(self.elements_widget)
        self.elements_widget.itemDoubleClicked.connect(self._on_click_item)
        self.coordinate_system_widget.currentItemChanged.connect(
            lambda item: self.elements_widget._onItemChange(item.text())
        )
        self.coordinate_system_widget.currentItemChanged.connect(
            lambda item: self.coordinate_system_widget._select_coord_sys(item.text())
        )
        self.viewer_model.layer_saved.connect(self.elements_widget._onItemChange)
        self.coordinate_system_widget.currentItemChanged.connect(self._update_layers_visibility)
        self.coordinate_system_widget.currentItemChanged.connect(
            lambda item: self.viewer_model._affine_transform_layers(item.text())
        )
        self.viewer_model.viewer.layers.events.inserted.connect(self._on_insert_layer)

    def _on_insert_layer(self, event: Event) -> None:
        layer = event.value
        layer.events.visible.connect(self._update_visible_in_coordinate_system)

    def _on_click_item(self, item: QListWidgetItem) -> None:
        self._onClick(item.text())

    def _hide_slider(self) -> None:
        self.slider.setVisible(False)

    def _onClick(self, text: str) -> None:
        selected_cs = self.coordinate_system_widget._system
        if self.worker_thread.isRunning():
            show_info("Please wait for the current operation to finish.")
            return

        if selected_cs and self.elements_widget._elements:
            sdata, multi = _get_sdata_key(self._sdata, self.elements_widget._elements, text)
            if (type_ := self.elements_widget._elements[text]["element_type"]) not in {
                "labels",
                "images",
                "shapes",
                "points",
            }:
                return

            type_ = cast(str, type_)

            self.worker_thread.load_data(type_, text, sdata, selected_cs, multi)
            if not PROBLEMATIC_NUMPY_MACOS:
                self.slider.setVisible(True)

    def _update_visible_in_coordinate_system(self, event: Event) -> None:
        """Toggle active in the coordinate system metadata when changing visibility of layer."""
        metadata = event.source.metadata
        layer_active = metadata.get("_active_in_cs")
        selected_coordinate_system = self.coordinate_system_widget._system

        elements = self.elements_widget._elements
        element_name = metadata.get("name")
        if elements and element_name and element_name in elements:
            if selected_coordinate_system not in layer_active:
                layer_active.add(selected_coordinate_system)
            else:
                layer_active.remove(selected_coordinate_system)

    def _update_layers_visibility(self) -> None:
        """Toggle layer visibility dependent on presence in currently selected coordinate system."""
        elements = self.elements_widget._elements
        coordinate_system = self.coordinate_system_widget._system
        # No layer selected on first time coordinate system selection
        if self.viewer_model.viewer.layers:
            for layer in self.viewer_model.viewer.layers:
                element_name = layer.metadata.get("name")
                if element_name:
                    if elements and (
                        layer.name not in elements or element_name != elements[layer.name]["original_name"]
                    ):
                        layer.visible = False
                    elif layer.metadata["_active_in_cs"]:
                        layer.visible = True
                        # Prevent _update_visible_in_coordinate_system of invalid removal of coordinate system
                        layer.metadata["_active_in_cs"].add(coordinate_system)
                        layer.metadata["_current_cs"] = coordinate_system

    def _get_shapes(self, sdata: SpatialData, key: str, selected_cs: str, multi: bool) -> Shapes | Points:
        original_name = key[: key.rfind("_")] if multi else key

        if type(sdata.shapes[original_name].iloc[0].geometry) is shapely.geometry.point.Point:
            return self.viewer_model.get_sdata_circles(sdata, key, selected_cs, multi)
        if (type(sdata.shapes[original_name].iloc[0].geometry) is shapely.geometry.polygon.Polygon) or (
            type(sdata.shapes[original_name].iloc[0].geometry) is shapely.geometry.multipolygon.MultiPolygon
        ):
            return self.viewer_model.get_sdata_shapes(sdata, key, selected_cs, multi)

        raise TypeError("Incorrect data type passed for shapes (should be Shapely Point or Polygon or MultiPolygon).")
