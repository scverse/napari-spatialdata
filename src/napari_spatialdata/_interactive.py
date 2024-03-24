from __future__ import annotations

from typing import TYPE_CHECKING, Any

import napari
from napari.utils.events import EventedList

from napari_spatialdata._sdata_widgets import SdataWidget
from napari_spatialdata.utils._utils import (
    NDArrayA,
    get_duplicate_element_names,
    get_elements_meta_mapping,
    get_itemindex_by_text,
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

    def add_element(self, element: str, element_coordinate_system: str, view_element_system: bool = False) -> None:
        """
        Add an element of a spatial data object being visualized with interactive to the viewer.

        Parameters
        ----------
        element
            The name of the element in the spatial data object to add as napari layer.
        element_coordinate_system
            The coordinate system in which the layer should be visualized.
        view_element_system
            Whether to switch to element_coordinate_system or to switch back to current active coordinate system
            after adding an element as layer.
        """
        duplicate_element_names, _ = get_duplicate_element_names(self._sdata)
        elements, name_to_add = get_elements_meta_mapping(
            self._sdata, element_coordinate_system, duplicate_element_names, element
        )
        if name_to_add:
            if view_element_system:
                widget_item = get_itemindex_by_text(
                    self._sdata_widget.coordinate_system_widget, element_coordinate_system
                )
                self._sdata_widget.coordinate_system_widget.setCurrentItem(widget_item)
                self._sdata_widget._onClick(name_to_add)
            else:
                cache_elements = self._sdata_widget.elements_widget._elements
                cache_coordinate_system = self._sdata_widget.coordinate_system_widget._system
                self._sdata_widget.elements_widget._elements = elements
                self._sdata_widget.coordinate_system_widget._system = element_coordinate_system
                self._sdata_widget._onClick(name_to_add)
                self._sdata_widget.elements_widget._elements = cache_elements
                self._sdata_widget.coordinate_system_widget._system = cache_coordinate_system
                self._viewer.layers[-1].visible = True
        else:
            raise ValueError(f"Element {element} not found in coordinate system {element_coordinate_system}.")

    def switch_coordinate_system(self, coordinate_system: str) -> None:
        """Switch to a coordinate system present in the spatialdata object(s)."""
        widget_item = get_itemindex_by_text(self._sdata_widget.coordinate_system_widget, coordinate_system)
        if widget_item:
            self._sdata_widget.coordinate_system_widget.setCurrentItem(widget_item)
        else:
            raise ValueError(f"{coordinate_system} not present as coordinate system in any of the spatialdata objects")

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
        """Run the napari application."""
        napari.run()

    def screenshot(self) -> NDArrayA | Any:
        """Take a screenshot of the viewer in its current state."""
        return self._viewer.screenshot(canvas_only=False)
