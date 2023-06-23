from napari.viewer import Viewer
from napari_spatialdata._sdata_widgets import SdataWidget
from napari_spatialdata.utils._test_utils import click_list_widget_item, get_center_pos_listitem
from spatialdata.datasets import blobs

sdata = blobs(extra_coord_system="space")


def test_layer_visibility(qtbot, make_napari_viewer: any):
    viewer = make_napari_viewer()
    widget = SdataWidget(viewer, sdata)

    # Click on `space` coordinate system
    center_pos = get_center_pos_listitem(widget.coordinate_system_widget, "space")
    click_list_widget_item(qtbot, widget.coordinate_system_widget, center_pos, "currentItemChanged")

    widget._onClick(list(sdata.images.keys())[0])
    viewer.add_points()
    new_metadata = viewer.layers[-1].metadata
    assert new_metadata["_current_cs"] == "space"
    assert new_metadata["_active_in_cs"] == {"space"}
    assert viewer.layers[0].metadata["sdata"] == new_metadata["sdata"]

    Viewer.close_all()
