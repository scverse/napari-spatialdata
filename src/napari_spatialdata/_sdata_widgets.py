from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Iterable

import shapely
from napari.utils.events import EventedList
from qtpy.QtWidgets import QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget
from spatialdata import SpatialData

from napari_spatialdata._viewer import SpatialDataViewer
from napari_spatialdata.utils._utils import _get_sdata_key

if TYPE_CHECKING:
    from napari import Viewer
    from napari.utils.events.event import Event


class ElementWidget(QListWidget):
    def __init__(self, sdata: EventedList):
        super().__init__()
        self._sdata = sdata

    def _onClickChange(self, selected_coordinate_system: QListWidgetItem | int | Iterable[str]) -> None:
        self.clear()

        elements = {}
        element_names = [element_name for sdata in self._sdata for _, element_name, _ in sdata._gen_elements()]
        element_names = [element for element, count in Counter(element_names).items() if count > 1]

        for index, sdata in enumerate(self._sdata):
            for element_type, element_name, _ in sdata.filter_by_coordinate_system(
                selected_coordinate_system
            )._gen_elements():
                # This allows us to handle SpatialElement with the same name in different SpatialData objects
                if element_name not in element_names:
                    elements[element_name] = {
                        "element_type": element_type,
                        "sdata_index": index,
                        "original_name": element_name,
                    }
                else:
                    new_name = element_name + f"_{index}"
                    elements[new_name] = {
                        "element_type": element_type,
                        "sdata_index": index,
                        "original_name": element_name,
                    }

        self.addItems(elements.keys())
        self._elements = elements


class CoordinateSystemWidget(QListWidget):
    def __init__(self, sdata: EventedList):
        super().__init__()

        self._sdata = sdata

        coordinate_systems = {cs for sdata in self._sdata for cs in sdata.coordinate_systems}
        self.addItems(coordinate_systems)

    def _select_coord_sys(self, selected_coordinate_system: QListWidgetItem | int | Iterable[str]) -> None:
        self._system = str(selected_coordinate_system)


class SdataWidget(QWidget):
    def __init__(self, viewer: Viewer, sdata: EventedList):
        super().__init__()
        self._sdata = sdata
        self.viewer_model = SpatialDataViewer(viewer, self._sdata)

        self.setLayout(QVBoxLayout())

        self.coordinate_system_widget = CoordinateSystemWidget(self._sdata)
        self.elements_widget = ElementWidget(self._sdata)

        self.layout().addWidget(QLabel("Coordinate System:"))
        self.layout().addWidget(self.coordinate_system_widget)
        self.layout().addWidget(QLabel("Elements:"))
        self.layout().addWidget(self.elements_widget)
        self.elements_widget.itemDoubleClicked.connect(lambda item: self._onClick(item.text()))
        self.coordinate_system_widget.itemClicked.connect(lambda item: self.elements_widget._onClickChange(item.text()))
        self.coordinate_system_widget.itemClicked.connect(
            lambda item: self.coordinate_system_widget._select_coord_sys(item.text())
        )
        self.coordinate_system_widget.itemClicked.connect(self._update_layers_visibility)
        self.viewer_model.viewer.layers.events.inserted.connect(self._on_insert_layer)

    def _on_insert_layer(self, event: Event) -> None:
        layer = event.value
        layer.events.visible.connect(self._update_visible_in_coordinate_system)

    def _onClick(self, text: str) -> None:
        if self.elements_widget._elements[text]["element_type"] == "labels":
            self._add_label(text)
        elif self.elements_widget._elements[text]["element_type"] == "images":
            self._add_image(text)
        elif self.elements_widget._elements[text]["element_type"] == "points":
            self._add_points(text)
        elif self.elements_widget._elements[text]["element_type"] == "shapes":
            self._add_shapes(text)

    def _update_visible_in_coordinate_system(self, event: Event) -> None:
        """Toggle active in the coordinate system metadata when changing visibility of layer."""
        layer = event.source
        layer_active = layer.metadata["_active_in_cs"]
        selected_coordinate_system = self.coordinate_system_widget._system

        elements = self.elements_widget._elements
        if layer.name in elements:
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
                if "sdata" in layer.metadata:
                    if layer.name not in elements:
                        layer.visible = False
                    elif layer.metadata["_active_in_cs"]:
                        layer.visible = True
                        # Prevent _update_visible_in_coordinate_system of invalid removal of coordinate system
                        layer.metadata["_active_in_cs"].add(coordinate_system)
                        layer.metadata["_current_cs"] = coordinate_system

    def _add_circles(self, sdata: SpatialData, key: str, multi: bool) -> None:
        selected_cs = self.coordinate_system_widget._system
        self.viewer_model.add_sdata_circles(sdata, selected_cs, key, multi)

    def _add_polygons(self, sdata: SpatialData, key: str, multi: bool) -> None:
        selected_cs = self.coordinate_system_widget._system
        self.viewer_model.add_sdata_shapes(sdata, selected_cs, key, multi)

    def _add_shapes(self, key: str) -> None:
        sdata, multi = _get_sdata_key(self._sdata, self.elements_widget._elements, key)

        original_name = key
        if multi:
            original_name = original_name[: original_name.rfind("_")]

        if type(sdata.shapes[original_name].iloc[0][0]) == shapely.geometry.point.Point:
            self._add_circles(sdata, key, multi)
        elif (type(sdata.shapes[original_name].iloc[0][0]) == shapely.geometry.polygon.Polygon) or (
            type(sdata.shapes[original_name].iloc[0][0]) == shapely.geometry.multipolygon.MultiPolygon
        ):
            self._add_polygons(sdata, key, multi)
        else:
            raise TypeError(
                "Incorrect data type passed for shapes (should be Shapely Point or Polygon or MultiPolygon)."
            )

    def _add_label(self, key: str) -> None:
        selected_cs = self.coordinate_system_widget._system
        sdata, multi = _get_sdata_key(self._sdata, self.elements_widget._elements, key)
        self.viewer_model.add_sdata_labels(sdata, selected_cs, key, multi)

    def _add_image(self, key: str) -> None:
        selected_cs = self.coordinate_system_widget._system
        sdata, multi = _get_sdata_key(self._sdata, self.elements_widget._elements, key)
        self.viewer_model.add_sdata_image(sdata, selected_cs, key, multi)

    def _add_points(self, key: str) -> None:
        selected_cs = self.coordinate_system_widget._system
        sdata, multi = _get_sdata_key(self._sdata, self.elements_widget._elements, key)
        self.viewer_model.add_sdata_points(sdata, selected_cs, key, multi)
