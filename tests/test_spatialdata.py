import logging
from typing import Any

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from dask.array.random import randint
from dask.dataframe import DataFrame as DaskDataFrame
from dask.dataframe import from_dask_array
from datatree import DataTree
from multiscale_spatial_image import to_multiscale
from napari.layers import Image, Labels, Points
from napari.utils.events import EventedList
from napari_spatialdata import QtAdataViewWidget
from napari_spatialdata._sdata_widgets import CoordinateSystemWidget, ElementWidget, SdataWidget
from napari_spatialdata.constants import config
from napari_spatialdata.utils._test_utils import click_list_widget_item, get_center_pos_listitem
from numpy import int64
from spatialdata import SpatialData, deepcopy
from spatialdata._core.query.relational_query import get_element_instances
from spatialdata.datasets import blobs
from spatialdata.models import PointsModel, TableModel
from spatialdata.transformations import Identity
from spatialdata.transformations.operations import set_transformation
from xarray import DataArray

RNG = np.random.default_rng(seed=0)


def test_elementwidget(make_napari_viewer: Any, blobs_extra_cs: SpatialData):
    _ = make_napari_viewer()
    widget = ElementWidget(EventedList([blobs_extra_cs]))
    assert widget._sdata is not None
    assert not widget._elements
    widget._onItemChange("global")
    assert widget._elements
    for name in blobs_extra_cs.images:
        assert widget._elements[name]["element_type"] == "images"
    for name in blobs_extra_cs.labels:
        assert widget._elements[name]["element_type"] == "labels"
    for name in blobs_extra_cs.points:
        assert widget._elements[name]["element_type"] == "points"
    for name in blobs_extra_cs.shapes:
        assert widget._elements[name]["element_type"] == "shapes"


def test_coordinatewidget(make_napari_viewer: Any, blobs_extra_cs: SpatialData):
    _ = make_napari_viewer()
    widget = CoordinateSystemWidget(EventedList([blobs_extra_cs]))
    items = [widget.item(x).text() for x in range(widget.count())]
    assert len(items) == len(blobs_extra_cs.coordinate_systems)
    for item in items:
        assert item in blobs_extra_cs.coordinate_systems


def test_sdatawidget_images(make_napari_viewer: Any, blobs_extra_cs: SpatialData):
    viewer = make_napari_viewer()
    widget = SdataWidget(viewer, EventedList([blobs_extra_cs]))
    assert len(widget.viewer_model.viewer.layers) == 0
    widget.coordinate_system_widget._select_coord_sys("global")
    widget.elements_widget._onItemChange("global")
    widget._onClick(list(blobs_extra_cs.images.keys())[0])
    assert len(widget.viewer_model.viewer.layers) == 1
    assert isinstance(widget.viewer_model.viewer.layers[0], Image)
    assert widget.viewer_model.viewer.layers[0].name == list(blobs_extra_cs.images.keys())[0]
    blobs_extra_cs.images["image"] = to_multiscale(blobs_extra_cs.images["blobs_image"], [2, 4])
    widget.elements_widget._onItemChange("global")
    widget._onClick("image")

    assert len(widget.viewer_model.viewer.layers) == 2
    assert (widget.viewer_model.viewer.layers[0].data == widget.viewer_model.viewer.layers[1].data._data[0]).all()
    del blobs_extra_cs.images["image"]


def test_sdatawidget_labels(make_napari_viewer: Any, blobs_extra_cs: SpatialData):
    viewer = make_napari_viewer()
    widget = SdataWidget(viewer, EventedList([blobs_extra_cs]))
    assert len(widget.viewer_model.viewer.layers) == 0
    widget.coordinate_system_widget._select_coord_sys("global")
    widget.elements_widget._onItemChange("global")
    widget._onClick(list(blobs_extra_cs.labels.keys())[0])
    assert len(widget.viewer_model.viewer.layers) == 1
    assert widget.viewer_model.viewer.layers[0].name == list(blobs_extra_cs.labels.keys())[0]
    assert isinstance(widget.viewer_model.viewer.layers[0], Labels)
    assert isinstance(widget.viewer_model.viewer.layers[0].metadata.get("adata"), AnnData)
    assert (
        widget.viewer_model.viewer.layers[0].metadata.get("adata").n_obs
        == (
            blobs_extra_cs["table"].obs[blobs_extra_cs["table"].uns["spatialdata_attrs"]["region_key"]]
            == list(blobs_extra_cs.labels.keys())[0]
        ).sum()
    )
    assert widget.viewer_model.viewer.layers[0].metadata.get("region_key") is not None


def test_sdatawidget_points(caplog, make_napari_viewer: Any, blobs_extra_cs: SpatialData):
    config.POINT_THRESHOLD = 400
    blobs_extra_cs.points["many_points"] = PointsModel.parse(
        from_dask_array(randint(0, 10, [800, 2], dtype=int64), columns=["x", "y"])
    )
    adata = AnnData(
        X=RNG.normal(size=(800, 1)),
        obs=pd.DataFrame({"instance_id": list(range(800)), "region": ["many_points"] * 800}),
    )
    table = TableModel.parse(adata, region_key="region", region="many_points", instance_key="instance_id")
    blobs_extra_cs["many_points_table"] = table

    # same length as blobs_points
    adata = AnnData(
        X=RNG.normal(size=(200, 2)),
        obs=pd.DataFrame({"instance_id": list(range(200)), "region": ["blobs_points"] * 200}),
    )
    table = TableModel.parse(adata, region_key="region", region="blobs_points", instance_key="instance_id")
    blobs_extra_cs["table"] = table
    set_transformation(blobs_extra_cs.points["many_points"], {"global": Identity()}, set_all=True)

    viewer = make_napari_viewer()
    widget = SdataWidget(viewer, EventedList([blobs_extra_cs]))
    assert len(widget.viewer_model.viewer.layers) == 0
    widget.coordinate_system_widget._select_coord_sys("global")
    widget.elements_widget._onItemChange("global")
    widget._onClick(list(blobs_extra_cs.points.keys())[0])
    assert len(widget.viewer_model.viewer.layers) == 1
    assert widget.viewer_model.viewer.layers[0].name == list(blobs_extra_cs.points.keys())[0]
    assert isinstance(widget.viewer_model.viewer.layers[0], Points)
    assert isinstance(widget.viewer_model.viewer.layers[0].metadata.get("adata"), AnnData)
    assert widget.viewer_model.viewer.layers[0].metadata.get("adata").n_obs == len(
        blobs_extra_cs.points["blobs_points"]["x"]
    )

    widget._onClick("many_points")
    with caplog.at_level(logging.INFO):
        assert (
            "Subsampling points because the number of points exceeds the currently supported 400."
            in caplog.records[0].message
        )
    assert widget.viewer_model.viewer.layers[1].metadata.get("adata").n_obs == 400
    del blobs_extra_cs.points["many_points"]


def test_layer_visibility(qtbot, make_napari_viewer: Any, blobs_extra_cs: SpatialData):
    # Only points layer in coordinate system `other`
    set_transformation(
        blobs_extra_cs.points[list(blobs_extra_cs.points.keys())[0]], Identity(), to_coordinate_system="other"
    )
    viewer = make_napari_viewer()
    widget = SdataWidget(viewer, EventedList([blobs_extra_cs]))

    # Click on `global` coordinate system
    center_pos = get_center_pos_listitem(widget.coordinate_system_widget, "global")
    click_list_widget_item(qtbot, widget.coordinate_system_widget, center_pos, "currentItemChanged")

    # Load 2 layers both are visible
    widget._onClick(list(blobs_extra_cs.points.keys())[0])
    widget._onClick(list(blobs_extra_cs.labels.keys())[0])

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


def test_multiple_sdata(qtbot, make_napari_viewer: Any, blobs_extra_cs: SpatialData):
    # Create additional sdata with one extra element that is unique
    sdata_mock = blobs(extra_coord_system="test")
    sdata_mock.points["extra"] = PointsModel.parse(
        from_dask_array(randint(0, 10, [5, 2], dtype=int64), columns=["x", "y"])
    )
    set_transformation(sdata_mock.points["extra"], {"global": Identity()}, set_all=True)

    viewer = make_napari_viewer()
    # qtbot.addWidget(viewer.window._qt_viewer)
    widget = SdataWidget(viewer, EventedList([blobs_extra_cs, sdata_mock]))

    # Click on `global` coordinate system
    center_pos = get_center_pos_listitem(widget.coordinate_system_widget, "global")
    click_list_widget_item(qtbot, widget.coordinate_system_widget, center_pos, "currentItemChanged")

    # _0 suffix for sdata and _1 for sdata_mock as it is based on index.
    widget._onClick(list(blobs_extra_cs.images.keys())[0] + "_0")
    widget._onClick(list(sdata_mock.images.keys())[0] + "_1")

    # Extra is unique and thus should not have suffix
    assert viewer.layers[0].metadata["sdata"] is blobs_extra_cs
    assert viewer.layers[1].metadata["sdata"] is sdata_mock
    assert "extra" in widget.elements_widget._elements

    # Only elements of sdata present in space
    center_pos = get_center_pos_listitem(widget.coordinate_system_widget, "space")
    click_list_widget_item(qtbot, widget.coordinate_system_widget, center_pos, "currentItemChanged")
    assert all(element_name.endswith("0") for element_name in list(widget.elements_widget._elements.keys()))

    widget._onClick(list(blobs_extra_cs.labels.keys())[0] + "_0")
    assert viewer.layers[-1].metadata["sdata"] is blobs_extra_cs

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


@pytest.mark.xfail
@pytest.mark.parametrize("instance_key_type", ["int", "str"])
def test_partial_table_matching_with_arbitrary_ordering(qtbot, make_napari_viewer: Any, instance_key_type: str):
    """
    Test plotting when the table has less or extra rows than the spatial elements, and the order is arbitrary.
    """
    # - Remove the original table, then add additional tables, one per spatial element, so that each element is
    #   annotated by a table.
    # - The table will have some rows referring to instances that are not present in the spatial elements, and will lack
    #   some rows referring to instances that are present in the spatial elements.
    # - At the same time we construct a parallel SpatialData object with permuted order both for the spatial element
    #   and for the table rows
    original_sdata = blobs()
    del original_sdata.tables["table"]
    shuffled_element_dicts = {}
    REGION_KEY = "region"
    INSTANCE_KEY = "instance_id"
    for region in [
        "blobs_labels",
        "blobs_multiscale_labels",
        "blobs_points",
        "blobs_circles",
        "blobs_multipolygons",
        "blobs_polygons",
    ]:
        element = original_sdata[region]
        if isinstance(element, (DataArray, DataTree)):
            index = get_element_instances(element).values
        elif isinstance(element, DaskDataFrame):
            index = element.index.compute().values
        else:
            index = element.index.values
        # remove one row, so that one instance in the spatial element is not present in the table
        adjusted_index = index[:-1]
        # add annotation to an instance that is not present in the spatial element
        adjusted_index = np.append(adjusted_index, [1000])
        table = AnnData(
            X=np.zeros((len(adjusted_index), 1)),
            obs={REGION_KEY: region, INSTANCE_KEY: adjusted_index, "annotation": np.arange(len(adjusted_index))},
        )
        table = TableModel.parse(table, region=region, region_key=REGION_KEY, instance_key=INSTANCE_KEY)
        table_name = region + "_table"
        original_sdata.tables[table_name] = table

        # when instance_key_type == 'str' (and when the element is not Labels), let's change the type of instance_key
        # column and of the corresponding index in the spatial element to string. Labels need to have int as they are
        # tensors of non-negative integers.
        if not isinstance(element, (DataArray, DataTree)) and instance_key_type == "str":
            element.index = element.index.astype(str)
            table.obs[INSTANCE_KEY] = table.obs[INSTANCE_KEY].astype(str)

        shuffled_element = deepcopy(element)
        shuffled_table = deepcopy(table)

        # shuffle the order of the rows of the element (when the element is not Labels)
        if not isinstance(element, (DataArray, DataTree)):
            shuffled_element = shuffled_element.loc[RNG.permutation(shuffled_element.index)]
        # shuffle the order of the rows of the table
        shuffled_table = shuffled_table[RNG.permutation(shuffled_table.obs.index), :].copy()

        shuffled_element_dicts[region] = shuffled_element
        shuffled_element_dicts[table_name] = shuffled_table
    ##
    shuffled_sdata = SpatialData.from_elements_dict(shuffled_element_dicts)

    # to manually check the results
    # from napari_spatialdata import Interactive
    # Interactive([original_sdata, shuffled_sdata])

    # TODO: here below we should make the test automatic: compare if the plot of annotation is the same for the original
    # and the shuffled sdata object
    viewer = make_napari_viewer()
    widget = SdataWidget(viewer, EventedList([original_sdata, shuffled_sdata]))
    viewer.window.add_dock_widget(widget, name="SpatialData")
    view_widget = QtAdataViewWidget(viewer)

    # Click on `global` coordinate system
    center_pos = get_center_pos_listitem(widget.coordinate_system_widget, "global")
    click_list_widget_item(qtbot, widget.coordinate_system_widget, center_pos, "currentItemChanged")

    widget._onClick(list(shuffled_sdata.points.keys())[0] + "_1")
    view_widget._select_layer()

    center_pos = get_center_pos_listitem(view_widget.obs_widget, "instance_id")
    click_list_widget_item(qtbot, view_widget.obs_widget, center_pos, "itemDoubleClicked", click="double")
