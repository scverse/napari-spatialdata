from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import napari
import pandas as pd
from loguru import logger
from napari.utils.events import EventedList

from napari_spatialdata._sdata_widgets import SdataWidget
from napari_spatialdata.utils._utils import (
    NDArrayA,
    get_duplicate_element_names,
    get_elements_meta_mapping,
)

if TYPE_CHECKING:
    from spatialdata import SpatialData
import matplotlib.pyplot as plt
from spatialdata._core.query.relational_query import get_values
from spatialdata.models import TableModel

from napari_spatialdata.utils._test_utils import create_generated_screenshots_folder, save_image


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

    def get_random_subset_of_columns(self, coordinate_system_name: str) -> pd.Dataframe:
        annotation_element = self._sdata.table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY]
        annotation_key = self._sdata.table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]

        for element_type, element_name, _ in self._sdata.filter_by_coordinat_system(
            coordinate_system_name
        )._gen_elements():
            if element_name == annotation_element:
                if element_type == "images":
                    # No annotation
                    return None
                if element_type == "labels":
                    # Retrieve table annotation
                    pass
                elif element_type == "points" or element_type == "shapes":
                    v = get_values(value_key=annotation_key, sdata=self._sdata, element_name=annotation_element)

            return v

        return None

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

    def __init__(
        self,
        sdata: SpatialData,
        coordinate_system_name: str | None = None,
        headless: bool = False,
        _test_notebook_name: str | None = None,
        _notebook_cell_id: str | None = None,
        _generate_screenshots: str | None = None,
    ) -> None:
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

        if _test_notebook_name is not None:
            assert _notebook_cell_id is not None

            # Select the first coordiante system only
            coordinate_system_name = str(self._sdata[0].coordinate_systems[0])
            logger.debug(f"Coordinate system selected for testing: {coordinate_system_name}")

            for _, element_name, _ in (
                self._sdata[0].filter_by_coordinate_system(coordinate_system_name)._gen_elements()
            ):
                self.add_element(coordinate_system_name=coordinate_system_name, element=element_name)
                # self.get_random_subset_of_columns(coordinate_system_name=coordinate_system_name)

                filepath = create_generated_screenshots_folder(_test_notebook_name, _notebook_cell_id)

                if _generate_screenshots:
                    save_image(self.screenshot(canvas_only=True), os.path.join(filepath, element_name + ".png"))
                else:
                    plt.imshow(self.screenshot(canvas_only=True))

        if not headless:
            self.run()

    def run(self) -> None:
        napari.run()

    def screenshot(self, canvas_only: bool) -> NDArrayA | Any:
        return self._viewer.screenshot(canvas_only=canvas_only)
