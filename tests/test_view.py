from typing import Any

import pytest
from napari.utils.events import EventedList
from napari_spatialdata._sdata_widgets import SdataWidget
from napari_spatialdata._view import QtAdataViewWidget
from spatialdata.datasets import blobs


@pytest.mark.parametrize("widget", [QtAdataViewWidget])
@pytest.mark.parametrize("n_channels", [3, 5])
def test_channel_slider_images(qtbot, make_napari_viewer: any, widget: Any, n_channels: int):
    channels = [f"channel_{i}" for i in range(n_channels)]
    sdata_blobs = blobs(c_coords=channels)
    viewer = make_napari_viewer()
    sdata_widget = SdataWidget(viewer, EventedList([sdata_blobs]))

    viewer.window.add_dock_widget(sdata_widget, name="SpatialData")
    sdata_widget.viewer_model.add_sdata_image(sdata_blobs, "blobs_image", "global", False)

    # this connects the slider to the viewer (done in __init__)
    _ = widget(viewer)

    # check if the slider is present
    start, stop, step = viewer.dims.range[0]
    assert start == 0
    assert stop == n_channels - 1
    assert step == 1

    # simulate position change of the slider
    viewer.dims.set_current_step(0, 0)
    qtbot.wait(50)  # wait for a short time to simulate user interaction
    viewer.dims.set_current_step(0, 1)
    qtbot.wait(50)  # wait for a short time to simulate user interaction

    viewer.close()
