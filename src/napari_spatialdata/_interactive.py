from __future__ import annotations

from typing import TYPE_CHECKING, Any

import napari

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
