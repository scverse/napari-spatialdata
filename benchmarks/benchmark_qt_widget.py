import numpy as np
from napari import Viewer
from napari.layers import Image
from napari.utils.events import EventedList

from napari_spatialdata._sdata_widgets import SdataWidget


class SdataWidgetSuite:
    def setup(self) -> None:
        self.viewer = Viewer()
        self.evented_list = EventedList()
        self.widget = SdataWidget(self.viewer, self.evented_list)
        self.image = Image(np.zeros((10, 10)))

    def time_create_widget(self) -> None:
        SdataWidget(self.viewer, self.evented_list)

    def time_layer_added(self) -> None:
        self.viewer.add_layer(self.image)

    def teardown(self) -> None:
        self.viewer.close()
