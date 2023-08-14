from __future__ import annotations

from typing import TYPE_CHECKING, Any

import napari
import shapely

from napari_spatialdata._sdata_widgets import SdataWidget
from napari_spatialdata.utils._utils import NDArrayA

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

    def add_element(self, coordinate_system_name: str, element: str) -> SpatialData:
        for element_type, element_name, _ in self._sdata.filter_by_coordinate_system(
            coordinate_system_name
        )._gen_elements():
            if element_name == element:
                if element_type == "images":
                    self._sdata_widget.viewer_model.add_sdata_image(self._sdata, coordinate_system_name, element)
                elif element_type == "labels":
                    self._sdata_widget.viewer_model.add_sdata_labels(self._sdata, coordinate_system_name, element)
                elif element_type == "points":
                    self._sdata_widget.viewer_model.add_sdata_points(self._sdata, coordinate_system_name, element)
                elif element_type == "shapes":
                    if type(self._sdata.shapes[element].iloc[0][0]) == shapely.geometry.point.Point:
                        self._sdata_widget.viewer_model.add_sdata_circles(self._sdata, coordinate_system_name, element)
                    elif (type(self._sdata.shapes[element].iloc[0][0]) == shapely.geometry.polygon.Polygon) or (
                        type(self._sdata.shapes[element].iloc[0][0]) == shapely.geometry.multipolygon.MultiPolygon
                    ):
                        self._sdata_widget.viewer_model.add_sdata_shapes(self._sdata, coordinate_system_name, element)

                break

    def __init__(self, sdata: SpatialData, headless: bool = False) -> None:
        viewer = napari.current_viewer()
        self._viewer = viewer if viewer else napari.Viewer()
        self._sdata = sdata
        self._sdata_widget = SdataWidget(self._viewer, sdata)
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
