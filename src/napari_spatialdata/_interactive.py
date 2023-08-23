from __future__ import annotations

from typing import TYPE_CHECKING, Any

import napari
import shapely
import os

from napari_spatialdata._sdata_widgets import SdataWidget
from napari_spatialdata.utils._utils import NDArrayA

if TYPE_CHECKING:
    from spatialdata import SpatialData
import matplotlib.pyplot as plt
from napari_spatialdata.utils._test_utils import save_image, take_screenshot


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
        else:
            raise ValueError("Element {element_name} not found in coordinate system {coordinate_system_name}.")
        

    def create_folders(self, tested_notebook, test_target):
        
        main_folder = os.getcwd()  # Get the current working directory
        tests_folder = os.path.join(main_folder, "tests")
        notebook_folder = os.path.join(tests_folder, tested_notebook)
        cell_folder = os.path.join(notebook_folder, test_target)

        print("Creating folders: !!!")
        if not os.path.exists(tests_folder):
            os.mkdir(tests_folder)

        if not os.path.exists(notebook_folder):
            os.mkdir(notebook_folder)

        if not os.path.exists(cell_folder):
            os.mkdir(cell_folder)

        return cell_folder


    def __init__(self, sdata: SpatialData, tested_notebook: str | None = None, test_target: str| None = None, take_screenshot: bool | None = None , headless: bool = False) -> None:
       
            
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

        if tested_notebook is not None:

            filepath = self.create_folders(tested_notebook, test_target)
            fig, axs = plt.subplots(ncols=3, figsize=(12, 3))

            for element_type, element_name, _ in self._sdata.filter_by_coordinate_system(
            "global"
            )._gen_elements():
                
                self.add_element(coordinate_system_name="global", element=element_name)

                axis_count = 0

                if take_screenshot:
                    save_image(self.screenshot(canvas_only=True), filepath + "/" + element_name + ".png")
                else:
                    plt.imshow(self.screenshot(canvas_only=True))
                    axis_count += 1


    def run(self) -> None:
        napari.run()

    def screenshot(self, canvas_only: bool) -> NDArrayA | Any:
        return self._viewer.screenshot(canvas_only=canvas_only)
