from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import shapely
from napari.utils.events import EventedList
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget
from spatialdata import SpatialData

from napari_spatialdata._viewer import SpatialDataViewer
from napari_spatialdata.constants.config import N_CIRCLES_WARNING_THRESHOLD, N_SHAPES_WARNING_THRESHOLD
from napari_spatialdata.utils._utils import _get_sdata_key, get_duplicate_element_names, get_elements_meta_mapping

if TYPE_CHECKING:
    from napari import Viewer
    from napari.utils.events.event import Event

icon_path = Path(__file__).parent / "resources/exclamation.png"


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
        for key, dict_val in elements.items():
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

    def _onClick(self, text: str) -> None:
        selected_cs = self.coordinate_system_widget._system

        if selected_cs and self.elements_widget._elements:
            sdata, multi = _get_sdata_key(self._sdata, self.elements_widget._elements, text)
            if self.elements_widget._elements[text]["element_type"] == "labels":
                self.viewer_model.add_sdata_labels(sdata, text, selected_cs, multi)
            elif self.elements_widget._elements[text]["element_type"] == "images":
                self.viewer_model.add_sdata_image(sdata, text, selected_cs, multi)
            elif self.elements_widget._elements[text]["element_type"] == "points":
                self.viewer_model.add_sdata_points(sdata, text, selected_cs, multi)
            elif self.elements_widget._elements[text]["element_type"] == "shapes":
                self._add_shapes(sdata, text, selected_cs, multi)

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

    def _add_shapes(self, sdata: SpatialData, key: str, selected_cs: str, multi: bool) -> None:
        original_name = key[: key.rfind("_")] if multi else key

        if type(sdata.shapes[original_name].iloc[0].geometry) is shapely.geometry.point.Point:
            self.viewer_model.add_sdata_circles(sdata, key, selected_cs, multi)
        elif (type(sdata.shapes[original_name].iloc[0].geometry) is shapely.geometry.polygon.Polygon) or (
            type(sdata.shapes[original_name].iloc[0].geometry) is shapely.geometry.multipolygon.MultiPolygon
        ):
            self.viewer_model.add_sdata_shapes(sdata, key, selected_cs, multi)
        else:
            raise TypeError(
                "Incorrect data type passed for shapes (should be Shapely Point or Polygon or MultiPolygon)."
            )
