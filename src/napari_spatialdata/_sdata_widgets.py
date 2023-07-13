from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import shapely
from qtpy.QtWidgets import QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget
from spatialdata import SpatialData

from napari_spatialdata._viewer import SpatialDataViewer

if TYPE_CHECKING:
    from napari import Viewer
    from napari.utils.events.event import Event


class ElementWidget(QListWidget):
    def __init__(self, sdata: SpatialData):
        super().__init__()
        self._sdata = sdata

    def _onClickChange(self, selected_coordinate_system: QListWidgetItem | int | Iterable[str]) -> None:
        self.clear()

        elements = {}
        for element_type, element_name, _ in self._sdata.filter_by_coordinate_system(
            selected_coordinate_system
        )._gen_elements():
            elements[element_name] = element_type

        self.addItems(elements.keys())
        self._elements = elements


class CoordinateSystemWidget(QListWidget):
    def __init__(self, sdata: SpatialData):
        super().__init__()

        self._sdata = sdata

        self.addItems(self._sdata.coordinate_systems)

    def _select_coord_sys(self, selected_coordinate_system: QListWidgetItem | int | Iterable[str]) -> None:
        self._system = str(selected_coordinate_system)


class SdataWidget(QWidget):
    def __init__(self, viewer: Viewer, sdata: SpatialData):
        super().__init__()
        self._sdata = sdata
        self.viewer_model = SpatialDataViewer(viewer)

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
        if self.elements_widget._elements[text] == "labels":
            self._add_label(text)
        elif self.elements_widget._elements[text] == "images":
            self._add_image(text)
        elif self.elements_widget._elements[text] == "points":
            self._add_points(text)
        elif self.elements_widget._elements[text] == "shapes":
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
                if layer.name not in elements:
                    layer.visible = False
                elif layer.metadata["_active_in_cs"]:
                    layer.visible = True
                    # Prevent _update_visible_in_coordinate_system of invalid removal of coordinate system
                    layer.metadata["_active_in_cs"].add(coordinate_system)
                    layer.metadata["_current_cs"] = coordinate_system

    def _add_circles(self, key: str) -> None:
        selected_cs = self.coordinate_system_widget._system
        self.viewer_model.add_sdata_circles(self._sdata, selected_cs, key)

    def _add_polygons(self, key: str) -> None:
        selected_cs = self.coordinate_system_widget._system
        self.viewer_model.add_sdata_shapes(self._sdata, selected_cs, key)

    def _add_shapes(self, key: str) -> None:
        if type(self._sdata.shapes[key].iloc[0][0]) == shapely.geometry.point.Point:
            self._add_circles(key)
        elif (type(self._sdata.shapes[key].iloc[0][0]) == shapely.geometry.polygon.Polygon) or (
            type(self._sdata.shapes[key].iloc[0][0]) == shapely.geometry.multipolygon.MultiPolygon
        ):
            self._add_polygons(key)
        else:
            raise TypeError(
                "Incorrect data type passed for shapes (should be Shapely Point or Polygon or MultiPolygon)."
            )

    def _add_label(self, key: str) -> None:
        selected_cs = self.coordinate_system_widget._system
        self.viewer_model.add_sdata_labels(self._sdata, selected_cs, key)

    def _add_image(self, key: str) -> None:
        selected_cs = self.coordinate_system_widget._system
        self.viewer_model.add_sdata_image(self._sdata, selected_cs, key)

    def _add_points(self, key: str) -> None:
        selected_cs = self.coordinate_system_widget._system
        self.viewer_model.add_sdata_points(self._sdata, selected_cs, key)
