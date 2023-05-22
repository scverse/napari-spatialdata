from typing import Any, Union

import napari
from spatialdata import SpatialData

from napari_spatialdata._sdata_widgets import SdataWidget
from napari_spatialdata._utils import NDArrayA


class Interactive:
    def __init__(
        self, sdata: SpatialData, images: bool = False, labels: bool = False, shapes: bool = False, points: bool = False
    ):
        self._viewer = napari.Viewer()
        self._sdata = sdata
        self._sdata_widget = SdataWidget(self._viewer, sdata, images, labels, shapes, points)
        self._list_widget = self._viewer.window.add_dock_widget(
            self._sdata_widget, name="SpatialData", area="left", menu=self._viewer.window.window_menu
        )

    def run(self) -> None:
        napari.run()

    def screenshot(self) -> Union[NDArrayA, Any]:
        return self._viewer.screenshot(canvas_only=False)
