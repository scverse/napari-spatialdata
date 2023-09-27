import numpy as np
import pytest
from napari.utils.events import EventedList
from napari_spatialdata._sdata_widgets import SdataWidget
from napari_spatialdata.utils._test_utils import click_list_widget_item, get_center_pos_listitem
from napari_spatialdata.utils._utils import _get_transform
from spatialdata.datasets import blobs
from spatialdata.transformations import Translation, set_transformation

sdata = blobs(extra_coord_system="space")


def test_metadata_inheritance(qtbot, make_napari_viewer: any):
    viewer = make_napari_viewer()
    widget = SdataWidget(viewer, EventedList([sdata]))

    # Click on `global` coordinate system
    center_pos = get_center_pos_listitem(widget.coordinate_system_widget, "global")
    click_list_widget_item(qtbot, widget.coordinate_system_widget, center_pos, "currentItemChanged")

    widget._onClick(list(sdata.images.keys())[0])
    widget._onClick(list(sdata.images.keys())[1])
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


@pytest.mark.skip(reason="Currently the events.blocker does not work when testing like this.")
def test_layer_names_duplicates(qtbot, make_napari_viewer: any):
    viewer = make_napari_viewer()
    widget = SdataWidget(viewer, EventedList([sdata]))

    # Click on `global` coordinate system
    center_pos = get_center_pos_listitem(widget.coordinate_system_widget, "global")
    click_list_widget_item(qtbot, widget.coordinate_system_widget, center_pos, "currentItemChanged")

    image_name = list(sdata.images.keys())[0]
    label_name = list(sdata.labels.keys())[0]
    widget._add_image(image_name)
    widget._add_label(label_name)

    assert widget.viewer_model.layer_names == {image_name, label_name}

    widget.viewer_model.viewer.layers[1].name = image_name
    assert widget.viewer_model.viewer.layers[1].name == label_name


def test_layer_transform(qtbot, make_napari_viewer: any):
    set_transformation(
        sdata["blobs_image"], transformation=Translation([25, 50], axes=("y", "x")), to_coordinate_system="translate"
    )
    viewer = make_napari_viewer()
    widget = SdataWidget(viewer, EventedList([sdata]))

    # Click on `global` coordinate system
    center_pos = get_center_pos_listitem(widget.coordinate_system_widget, "global")
    click_list_widget_item(qtbot, widget.coordinate_system_widget, center_pos, "currentItemChanged")

    widget._onClick(list(sdata.images.keys())[0])
    viewer.add_image(viewer.layers[0].data)

    no_transform = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    affine_transform = _get_transform(sdata[list(sdata.images.keys())[0]], "translate")
    assert np.array_equal(viewer.layers[0].affine.affine_matrix, no_transform)
    assert np.array_equal(viewer.layers[1].affine.affine_matrix, no_transform)

    # Click on `translate` coordinate system
    center_pos = get_center_pos_listitem(widget.coordinate_system_widget, "translate")
    click_list_widget_item(qtbot, widget.coordinate_system_widget, center_pos, "currentItemChanged")

    assert np.array_equal(viewer.layers[0].affine.affine_matrix, affine_transform)
    assert np.array_equal(viewer.layers[1].affine.affine_matrix, no_transform)
