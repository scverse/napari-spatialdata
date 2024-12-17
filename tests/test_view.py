from typing import Any

import numpy as np
import pytest
from napari.utils.events import EventedList
from spatialdata.datasets import blobs
from spatialdata.transformations import Affine, set_transformation

from napari_spatialdata._sdata_widgets import SdataWidget
from napari_spatialdata._view import QtAdataViewWidget
from napari_spatialdata.utils._test_utils import click_list_widget_item, get_center_pos_listitem


@pytest.mark.parametrize("widget", [QtAdataViewWidget])
@pytest.mark.parametrize("n_channels", [3, 5])
def test_channel_slider_images_bug_282(qtbot, make_napari_viewer: any, widget: Any, n_channels: int):
    # https://github.com/scverse/napari-spatialdata/issues/282
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
    assert step == 1

    # simulate position change of the slider
    viewer.dims.set_current_step(0, 0)
    qtbot.wait(50)  # wait for a short time to simulate user interaction
    viewer.dims.set_current_step(0, 1)
    qtbot.wait(50)  # wait for a short time to simulate user interaction

    viewer.close()


@pytest.mark.parametrize("widget", [QtAdataViewWidget])
@pytest.mark.parametrize("column", ["genes", "instance_id"])
def test_plot_dataframe_annotation_on_points(qtbot, make_napari_viewer: any, widget: Any, column: str):
    sdata_blobs = blobs()
    viewer = make_napari_viewer()
    sdata_widget = SdataWidget(viewer, EventedList([sdata_blobs]))

    viewer.window.add_dock_widget(sdata_widget, name="SpatialData")

    # init the adata view widget
    widget = widget(viewer)

    sdata_widget.viewer_model.add_sdata_points(sdata_blobs, "blobs_points", "global", False)

    # plot dataframe annotations on the points
    center_pos = get_center_pos_listitem(widget.dataframe_columns_widget, "instance_id")
    # TODO: the double click doesn't trigger the signal, so below we are calling _onAction directly (looking at a
    #  screenshot of the qtbot, the interface shows to be correctly clicked)
    click_list_widget_item(
        qtbot,
        widget=widget.dataframe_columns_widget,
        position=center_pos,
        wait_signal="currentItemChanged",
        click="double",
    )
    widget.dataframe_columns_widget._onAction([column])
    viewer.close()


@pytest.mark.parametrize("multiscale", [True, False])
@pytest.mark.parametrize("widget", [QtAdataViewWidget])
def test_channel_slider_images_bug_302(qtbot, make_napari_viewer: any, widget: Any, multiscale: bool):
    # https://github.com/scverse/napari-spatialdata/issues/302
    channels = [f"channel_{i}" for i in range(3)]
    sdata_blobs = blobs(c_coords=channels)

    # set a transformation to the single-scale and multi-scale images
    element_name = "blobs_image" if not multiscale else "blobs_multiscale_image"
    affine = Affine(np.array([[10, 20, 30], [40, 50, 60], [0, 0, 1]]), input_axes=("x", "y"), output_axes=("x", "y"))
    set_transformation(sdata_blobs[element_name], transformation=affine, to_coordinate_system="global")

    viewer = make_napari_viewer()
    sdata_widget = SdataWidget(viewer, EventedList([sdata_blobs]))

    viewer.window.add_dock_widget(sdata_widget, name="SpatialData")
    sdata_widget.viewer_model.add_sdata_image(sdata_blobs, element_name, "global", False)

    # this connects the slider to the viewer (done in __init__)
    _ = widget(viewer)

    # check if the slider is present
    start, stop, step = viewer.dims.range[0]
    assert start == 0
    assert step == 1

    # simulate position change of the slider
    viewer.dims.set_current_step(0, 0)
    qtbot.wait(50)  # wait for a short time to simulate user interaction
    viewer.dims.set_current_step(0, 1)
    qtbot.wait(50)  # wait for a short time to simulate user interaction

    viewer.close()
