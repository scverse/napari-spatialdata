from __future__ import annotations

from typing import TYPE_CHECKING, Any

import napari

from napari_spatialdata._sdata_widgets import SdataWidget
from napari_spatialdata._viewer import SpatialDataViewer
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

    Returns
    -------
    None
    """

    def __init__(self, sdata: SpatialData):
        self.sdata_viewer = SpatialDataViewer()
        # self._viewer = self.sdata_viewer._viewer
        self._sdata = sdata
        self._sdata_widget = SdataWidget(self.sdata_viewer, sdata)
        self._list_widget = self.sdata_viewer.window.add_dock_widget(
            self._sdata_widget, name="SpatialData", area="left", menu=self.sdata_viewer.window.window_menu
        )

    def run(self) -> None:
        napari.run()

    def screenshot(self) -> NDArrayA | Any:
        return self.sdata_viewer.screenshot(canvas_only=False)
