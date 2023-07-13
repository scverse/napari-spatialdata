from napari_spatialdata._sdata_widgets import SdataWidget
from napari_spatialdata.utils._test_utils import click_list_widget_item, get_center_pos_listitem
from spatialdata.datasets import blobs

sdata = blobs(extra_coord_system="space")


def test_metadata_inheritance(qtbot, make_napari_viewer: any):
    viewer = make_napari_viewer()
    widget = SdataWidget(viewer, sdata)

    # Click on `global` coordinate system
    center_pos = get_center_pos_listitem(widget.coordinate_system_widget, "global")
    click_list_widget_item(qtbot, widget.coordinate_system_widget, center_pos, "currentItemChanged")

    widget._add_image(list(sdata.images.keys())[0])
    widget._add_image(list(sdata.images.keys())[1])
    widget.viewer_model.viewer.add_shapes()

    # Two layers have the same spatialdata object. So we should count 1 spatialdata object.
    layers = widget.viewer_model.viewer.layers
    sdatas = [layer.metadata["sdata"] for layer in layers if "sdata" in layer.metadata]
    assert all(sdatas[0] is sdata for sdata in sdatas[1:])

    widget.viewer_model.inherit_metadata(widget.viewer_model.viewer.layers)

    # Now we did let the shapes layer inherit sdata from another layer. The number of unique spatialdata objects
    # should still be one.
    sdatas = [layer.metadata["sdata"] for layer in layers if "sdata" in layer.metadata]
    assert all(sdatas[0] is sdata for sdata in sdatas[1:])
    assert viewer.layers[-1].metadata["_current_cs"] == "global"
    assert viewer.layers[-1].metadata["_active_in_cs"] == {"global"}
