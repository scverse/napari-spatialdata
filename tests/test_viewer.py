import re
from pathlib import Path

import numpy as np
import pytest
from napari.utils.events import EventedList
from napari_spatialdata import QtAdataViewWidget
from napari_spatialdata._sdata_widgets import SdataWidget
from napari_spatialdata.utils._test_utils import click_list_widget_item, get_center_pos_listitem
from napari_spatialdata.utils._utils import _get_transform
from qtpy.QtCore import Qt
from spatialdata.datasets import blobs
from spatialdata.transformations import Scale, Translation, set_transformation

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
    qtbot.keyPress(viewer.window._qt_viewer, Qt.Key_L, Qt.ShiftModifier)

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
    viewer.close()


def test_adata_metadata(qtbot, make_napari_viewer: any):
    viewer = make_napari_viewer()
    widget = SdataWidget(viewer, EventedList([sdata]))
    viewer.window.add_dock_widget(widget, name="SpatialData")
    view_widget = QtAdataViewWidget(viewer)

    # Click on `global` coordinate system
    center_pos = get_center_pos_listitem(widget.coordinate_system_widget, "global")
    click_list_widget_item(qtbot, widget.coordinate_system_widget, center_pos, "currentItemChanged")

    widget._onClick("blobs_labels")
    assert viewer.layers[-1].metadata["adata"]
    assert view_widget.obs_widget.item(0)

    # Filtering adata leads to 0 rows so adata should be set to None
    widget._onClick("blobs_polygons")
    assert not viewer.layers[-1].metadata["adata"]
    assert not view_widget.obs_widget.item(0)
    viewer.close()


def test_save_layer(qtbot, tmp_path: str, make_napari_viewer: any):
    tmpdir = Path(tmp_path) / "tmp.zarr"
    sdata = blobs()
    sdata.write(tmpdir)
    first_shapes = list(sdata.shapes.keys())[0]
    # I transform to a different coordinate system to reproduce and test the bug described here:
    # https://github.com/scverse/napari-spatialdata/pull/168#issuecomment-1803080280
    set_transformation(
        sdata.shapes[first_shapes], transformation=Scale([2, 2], axes=("y", "x")), to_coordinate_system="global"
    )

    viewer = make_napari_viewer()
    widget = SdataWidget(viewer, EventedList([sdata]))

    # Click on `global` coordinate system
    center_pos = get_center_pos_listitem(widget.coordinate_system_widget, "global")
    click_list_widget_item(qtbot, widget.coordinate_system_widget, center_pos, "currentItemChanged")

    # Add an image layer
    widget._onClick(first_shapes)

    # --- test the shapes layer ---
    # Create a new shapes layer
    viewer.add_shapes()
    shapes_layer = viewer.layers[-1]

    with pytest.raises(ValueError, match="Cannot export a shapes element with no shapes"):
        widget.viewer_model._save_to_sdata(viewer)

    # add a polygon to the shapes layer
    shapes_layer.add([[[0, 0], [0, 1], [1, 1], [1, 0]]])

    # save the shapes layer to the sdata
    widget.viewer_model._save_to_sdata(viewer)
    assert "Shapes" in sdata.shapes

    # --- test the points layer ---
    viewer.add_points()
    points_layer = viewer.layers[-1]

    with pytest.raises(ValueError, match="Cannot export a points element with no points"):
        widget.viewer_model._save_to_sdata(viewer)

    # add a point to the points layer
    points_layer.add([0, 0])

    # save the points layer to the sdata
    widget.viewer_model._save_to_sdata(viewer)
    assert "Points" in sdata.points

    # Check overwriting element works
    layer_to_save = list(viewer.layers.selection)
    widget.viewer_model.save_to_sdata(layer_to_save, overwrite=True)

    # --- check the shapes and points layers got correctly saved ---
    # check that the elements widget got update with the new shapes and points elements
    n = len(widget.elements_widget)
    # I would have expected Shapes to be in position -2 and Points in position -1, but we have the following order
    # because elements of the same type are grouped together
    assert widget.elements_widget.item(n - 1).text() == "Shapes"
    assert widget.elements_widget.item(n - 5).text() == "Points"

    # add a new layer to the viewer with the newly saved shapes element
    widget._onClick("Shapes")
    new_shapes_layer = viewer.layers[-1]
    assert new_shapes_layer.name == "Shapes [1]"
    # I added the assert below to test against this bug here:
    # https://github.com/scverse/napari-spatialdata/pull/168#issuecomment-1803080280
    # which has been fixed by adding data_to_world() in save_to_sata().
    # In theory one would expect the following to fail if data_to_world() is removed from the lambda function called in
    # save_to_sdata(), but this doesn't happen, so this test is not covering that case.
    # In case the bug reappears let's try to make this test cover it, but for the moment let's not worry about it.
    assert np.array_equal(new_shapes_layer.data, [np.array([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])])

    # note that the added data doesn't close the polygon, this is ok
    assert shapes_layer.name == "Shapes"
    assert np.array_equal(shapes_layer.data, [np.array([[0, 0], [0, 1], [1, 1], [1, 0]])])

    # add a new layer to the viewer with the newly saved points element
    widget._onClick("Points")
    new_points_layer = viewer.layers[-1]
    assert new_points_layer.name == "Points [1]"
    assert np.array_equal(new_points_layer.data, [np.array([0, 0])])


def test_save_layer_no_sdata(qtbot, make_napari_viewer: any):
    viewer = make_napari_viewer()
    widget = SdataWidget(viewer, EventedList([]))

    viewer.add_shapes()
    shapes_layer = viewer.layers[-1]
    shapes_layer.add([[[0, 0], [0, 1], [1, 1], [1, 0]]])

    with pytest.raises(
        ValueError, match="No SpatialData layers found in the viewer. Layer cannot be linked to SpatialData object."
    ):
        widget.viewer_model._save_to_sdata(viewer)


def test_save_layer_multiple_selection(qtbot, tmp_path: str, make_napari_viewer: any):
    tmpdir = Path(tmp_path) / "tmp.zarr"
    tmpdir2 = Path(tmp_path) / "tmp2.zarr"
    sdata2 = blobs()
    sdata.write(tmpdir)
    sdata2.write(tmpdir2)
    viewer = make_napari_viewer()
    widget = SdataWidget(viewer, EventedList([sdata, sdata2]))

    # Click on `global` coordinate system
    center_pos = get_center_pos_listitem(widget.coordinate_system_widget, "global")
    click_list_widget_item(qtbot, widget.coordinate_system_widget, center_pos, "currentItemChanged")

    widget._onClick("blobs_image_0")
    widget._onClick("blobs_image_1")

    viewer.layers.selection.update({viewer.layers[0], viewer.layers[1]})

    viewer.add_shapes()
    shapes_layer = viewer.layers[-1]
    shapes_layer.add([[[0, 0], [0, 1], [1, 1], [1, 0]]])

    with pytest.raises(
        ValueError,
        match=(
            re.escape(
                "Multiple different spatialdata object found in the viewer. Please link the layer to one of them by "
                "selecting both the layer to save and the layer containing the SpatialData object and then pressing "
                "Shift+L. Then select the layer to save and press Shift+E again."
            )
        ),
    ):
        widget.viewer_model._save_to_sdata(viewer)

    # select the first image layer and the shapes layer and link them
    viewer.layers.selection.update({viewer.layers[0], viewer.layers[2]})
    qtbot.keyPress(viewer.window._qt_viewer, Qt.Key_L, Qt.ShiftModifier)

    with pytest.raises(ValueError, match="Only one layer can be saved at a time."):
        widget.viewer_model._save_to_sdata(viewer)

    # select the layer to save
    viewer.layers.selection = {viewer.layers[2]}
    # let's actually try the shortcut
    qtbot.keyPress(viewer.window._qt_viewer, Qt.Key_E, Qt.ShiftModifier)
    assert "Shapes" not in sdata2.shapes
    assert "Shapes" in sdata.shapes
