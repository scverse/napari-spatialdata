from __future__ import annotations

from typing import TYPE_CHECKING, Any

import napari
from napari.utils.events import EventedList

from napari_spatialdata._sdata_widgets import SdataWidget
from napari_spatialdata.utils._utils import (
    NDArrayA,
    get_duplicate_element_names,
    get_elements_meta_mapping,
)

if TYPE_CHECKING:
    from spatialdata import SpatialData


class Interactive:
    """
    Interactive visualization of spatial data.

    Parameters
    ----------
    sdata
        SpatialData object.
    headless
        Run napari in headless mode, default False.

    Returns
    -------
    None
    """

    def add_element(self, coordinate_system_name: str, element: str) -> None:
        duplicate_element_names, _ = get_duplicate_element_names(self._sdata)
        elements, name_to_add = get_elements_meta_mapping(
            self._sdata, coordinate_system_name, duplicate_element_names, element
        )
        if name_to_add:
            cache_elements = self._sdata_widget.elements_widget._elements
            cache_coordinate_system = self._sdata_widget.coordinate_system_widget._system
            self._sdata_widget.elements_widget._elements = elements
            self._sdata_widget.coordinate_system_widget._system = coordinate_system_name
            self._sdata_widget._onClick(name_to_add)
            self._sdata_widget.elements_widget._elements = cache_elements
            self._sdata_widget.coordinate_system_widget._system = cache_coordinate_system
        else:
            raise ValueError(f"Element {element} not found in coordinate system {coordinate_system_name}.")

    def __init__(self, sdata: SpatialData | list[SpatialData], headless: bool = False):
        viewer = napari.current_viewer()
        self._viewer = viewer if viewer else napari.Viewer()
        if isinstance(sdata, list):
            self._sdata = EventedList(data=sdata)
        else:
            self._sdata = EventedList(data=[sdata])
        self._sdata_widget = SdataWidget(self._viewer, self._sdata)
        self._list_widget = self._viewer.window.add_dock_widget(
            self._sdata_widget, name="SpatialData", area="left", menu=self._viewer.window.window_menu
        )
        self._viewer.window.add_plugin_dock_widget("napari-spatialdata", "View")
        if not headless:
            self.run()

    def run(self) -> None:
        napari.run()

    def screenshot(self) -> NDArrayA | Any:
        return self._viewer.screenshot(canvas_only=False)
