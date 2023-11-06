import logging
from typing import Any

import pytest
from anndata import AnnData
from dask.array.random import randint
from dask.dataframe import from_dask_array
from multiscale_spatial_image import to_multiscale
from napari.layers import Image, Labels, Points
from napari.utils.events import EventedList
from napari_spatialdata._sdata_widgets import CoordinateSystemWidget, ElementWidget, SdataWidget
from napari_spatialdata.utils._test_utils import click_list_widget_item, get_center_pos_listitem
from numpy import int64
from spatialdata.datasets import blobs
from spatialdata.transformations import Identity
from spatialdata.transformations.operations import set_transformation

sdata = blobs(extra_coord_system="space")


def test_elementwidget(make_napari_viewer: Any):
    _ = make_napari_viewer()
    widget = ElementWidget(EventedList([sdata]))
    assert widget._sdata is not None
    assert not widget._elements
    widget._onItemChange("global")
    assert widget._elements
    for name in sdata.images:
        assert widget._elements[name]["element_type"] == "images"
    for name in sdata.labels:
        assert widget._elements[name]["element_type"] == "labels"
    for name in sdata.points:
        assert widget._elements[name]["element_type"] == "points"
    for name in sdata.shapes:
        assert widget._elements[name]["element_type"] == "shapes"


def test_coordinatewidget(make_napari_viewer: Any):
    _ = make_napari_viewer()
    widget = CoordinateSystemWidget(EventedList([sdata]))
    items = [widget.item(x).text() for x in range(widget.count())]
    assert len(items) == len(sdata.coordinate_systems)
    for item in items:
        assert item in sdata.coordinate_systems


def test_sdatawidget_images(make_napari_viewer: Any):
    viewer = make_napari_viewer()
    widget = SdataWidget(viewer, EventedList([sdata]))
    assert len(widget.viewer_model.viewer.layers) == 0
    widget.coordinate_system_widget._select_coord_sys("global")
    widget.elements_widget._onItemChange("global")
    widget._onClick(list(sdata.images.keys())[0])
    assert len(widget.viewer_model.viewer.layers) == 1
    assert isinstance(widget.viewer_model.viewer.layers[0], Image)
    assert widget.viewer_model.viewer.layers[0].name == list(sdata.images.keys())[0]
    sdata.images["image"] = to_multiscale(sdata.images["blobs_image"], [2, 4])
    widget.elements_widget._onItemChange("global")
    widget._onClick("image")

    assert len(widget.viewer_model.viewer.layers) == 2
    assert (widget.viewer_model.viewer.layers[0].data == widget.viewer_model.viewer.layers[1].data._data[0]).all()
    del sdata.images["image"]


def test_sdatawidget_labels(make_napari_viewer: Any):
    viewer = make_napari_viewer()
    widget = SdataWidget(viewer, EventedList([sdata]))
    assert len(widget.viewer_model.viewer.layers) == 0
    widget.coordinate_system_widget._select_coord_sys("global")
    widget.elements_widget._onItemChange("global")
    widget._onClick(list(sdata.labels.keys())[0])
    assert len(widget.viewer_model.viewer.layers) == 1
    assert widget.viewer_model.viewer.layers[0].name == list(sdata.labels.keys())[0]
    assert isinstance(widget.viewer_model.viewer.layers[0], Labels)
    assert isinstance(widget.viewer_model.viewer.layers[0].metadata.get("adata"), AnnData)
    assert (
        widget.viewer_model.viewer.layers[0].metadata.get("adata").n_obs
        == (sdata.table.obs[sdata.table.uns["spatialdata_attrs"]["region_key"]] == list(sdata.labels.keys())[0]).sum()
    )
    assert widget.viewer_model.viewer.layers[0].metadata.get("region_key") is not None


def test_sdatawidget_points(caplog, make_napari_viewer: Any):
    sdata.points["many_points"] = from_dask_array(randint(0, 10, [200000, 2], dtype=int64), columns=["x", "y"])
    set_transformation(sdata.points["many_points"], {"global": Identity()}, set_all=True)

    viewer = make_napari_viewer()
    widget = SdataWidget(viewer, EventedList([sdata]))
    assert len(widget.viewer_model.viewer.layers) == 0
    widget.coordinate_system_widget._select_coord_sys("global")
    widget.elements_widget._onItemChange("global")
    widget._onClick(list(sdata.points.keys())[0])
    assert len(widget.viewer_model.viewer.layers) == 1
    assert widget.viewer_model.viewer.layers[0].name == list(sdata.points.keys())[0]
    assert isinstance(widget.viewer_model.viewer.layers[0], Points)
    assert isinstance(widget.viewer_model.viewer.layers[0].metadata.get("adata"), AnnData)
    assert widget.viewer_model.viewer.layers[0].metadata.get("adata").n_obs == len(sdata.points["blobs_points"]["x"])
    assert (
        len(widget.viewer_model.viewer.layers[0].metadata.get("adata").obs.keys())
        == sdata.points["blobs_points"].shape[1]
    )

    widget._onClick("many_points")
    with caplog.at_level(logging.INFO):
        assert (
            "Subsampling points because the number of points exceeds the currently supported 100 000."
            in caplog.records[0].message
        )
    assert widget.viewer_model.viewer.layers[1].metadata.get("adata").n_obs == 100000
    del sdata.points["many_points"]


def test_layer_visibility(qtbot, make_napari_viewer: Any):
    # Only points layer in coordinate system `other`
    set_transformation(sdata.points[list(sdata.points.keys())[0]], Identity(), to_coordinate_system="other")
    viewer = make_napari_viewer()
    widget = SdataWidget(viewer, EventedList([sdata]))

    # Click on `global` coordinate system
    center_pos = get_center_pos_listitem(widget.coordinate_system_widget, "global")
    click_list_widget_item(qtbot, widget.coordinate_system_widget, center_pos, "currentItemChanged")

    # Load 2 layers both are visible
    widget._onClick(list(sdata.points.keys())[0])
    widget._onClick(list(sdata.labels.keys())[0])

    points = viewer.layers[0]
    labels = viewer.layers[1]

    # Check that both are not an empty set
    assert points.metadata["_active_in_cs"]
    assert labels.metadata["_active_in_cs"]
    assert labels.metadata["_current_cs"] == "global"

    # Click on `space` coordinate system
    center_pos = get_center_pos_listitem(widget.coordinate_system_widget, "space")
    click_list_widget_item(qtbot, widget.coordinate_system_widget, center_pos, "currentItemChanged")

    # Is present in coordinate system and should stay visible.
    assert points.visible
    assert labels.visible
    assert points.metadata["_active_in_cs"] == {"global", "space"}
    assert labels.metadata["_active_in_cs"] == {"global", "space"}
    assert labels.metadata["_current_cs"] == "space"

    # Test visibility within same coordinate system
    labels.visible = False
    assert labels.metadata["_active_in_cs"] == {"global"}
    assert labels.metadata["_current_cs"] == "space"
    labels.visible = True

    # Click on `other` coordinate system
    center_pos = get_center_pos_listitem(widget.coordinate_system_widget, "other")
    click_list_widget_item(qtbot, widget.coordinate_system_widget, center_pos, "currentItemChanged")

    assert points.visible
    assert points.metadata["_active_in_cs"] == {"global", "space", "other"}
    assert not labels.visible
    # Since not present in current selected cs, this layer is still in previously selected cs.
    assert labels.metadata["_current_cs"] == "space"

    # Check case for landmark registration to make layer not in the coordinate system visible.
    labels.visible = True
    assert labels.metadata["_active_in_cs"] == {"global", "space"}

    # Check previously active coordinate system whether it is not removed.
    center_pos = get_center_pos_listitem(widget.coordinate_system_widget, "global")
    click_list_widget_item(qtbot, widget.coordinate_system_widget, center_pos, "currentItemChanged")

    assert points.visible
    assert points.metadata["_active_in_cs"] == {"global", "space", "other"}


def test_multiple_sdata(qtbot, make_napari_viewer: Any):
    # Create additional sdata with one extra element that is unique
    sdata_mock = blobs(extra_coord_system="test")
    sdata_mock.points["extra"] = from_dask_array(randint(0, 10, [5, 2], dtype=int64), columns=["x", "y"])
    set_transformation(sdata_mock.points["extra"], {"global": Identity()}, set_all=True)

    viewer = make_napari_viewer()
    # qtbot.addWidget(viewer.window._qt_viewer)
    widget = SdataWidget(viewer, EventedList([sdata, sdata_mock]))

    # Click on `global` coordinate system
    center_pos = get_center_pos_listitem(widget.coordinate_system_widget, "global")
    click_list_widget_item(qtbot, widget.coordinate_system_widget, center_pos, "currentItemChanged")

    # _0 suffix for sdata and _1 for sdata_mock as it is based on index.
    widget._onClick(list(sdata.images.keys())[0] + "_0")
    widget._onClick(list(sdata_mock.images.keys())[0] + "_1")

    # Extra is unique and thus should not have suffix
    assert viewer.layers[0].metadata["sdata"] is sdata
    assert viewer.layers[1].metadata["sdata"] is sdata_mock
    assert "extra" in widget.elements_widget._elements

    # Only elements of sdata present in space
    center_pos = get_center_pos_listitem(widget.coordinate_system_widget, "space")
    click_list_widget_item(qtbot, widget.coordinate_system_widget, center_pos, "currentItemChanged")
    assert all(element_name.endswith("0") for element_name in list(widget.elements_widget._elements.keys()))

    widget._onClick(list(sdata.labels.keys())[0] + "_0")
    assert viewer.layers[-1].metadata["sdata"] is sdata

    # Only elements of sdata present in test
    center_pos = get_center_pos_listitem(widget.coordinate_system_widget, "test")
    click_list_widget_item(qtbot, widget.coordinate_system_widget, center_pos, "currentItemChanged")
    assert all(element_name.endswith("1") for element_name in list(widget.elements_widget._elements.keys()))

    widget._onClick(list(sdata_mock.labels.keys())[0] + "_1")
    assert viewer.layers[-1].metadata["sdata"] is sdata_mock

    # test case of having empty layer selected and multiple sdata objects in viewer
    viewer.add_shapes()

    with pytest.raises(ValueError):
        widget.viewer_model._inherit_metadata(viewer)
        assert not viewer.layers[-1].metadata

    # test case of having multiple sdata objects in layer selection
    viewer.layers.select_all()
    with pytest.raises(ValueError):
        widget.viewer_model._inherit_metadata(viewer)
        assert not viewer.layers[-1].metadata

    viewer.layers.selection = viewer.layers[-2:]
    widget.viewer_model._inherit_metadata(viewer)
    assert viewer.layers[-1].metadata["sdata"] is sdata_mock
