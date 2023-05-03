import logging
from typing import Any

from anndata import AnnData
from dask.array.random import randint
from dask.dataframe import from_dask_array
from multiscale_spatial_image import to_multiscale
from napari.layers import Image, Labels, Points
from napari_spatialdata._interactive import CoordinateSystemWidget, ElementWidget, SdataWidget
from numpy import int64
from spatialdata.datasets import blobs
from spatialdata.transformations import Identity
from spatialdata.transformations.operations import set_transformation

sdata = blobs()


def test_elementwidget(make_napari_viewer: Any):
    _ = make_napari_viewer()
    widget = ElementWidget(sdata)
    assert widget._sdata is not None
    assert not hasattr(widget, "_elements")
    widget._onClickChange("global")
    assert hasattr(widget, "_elements")
    for name in sdata.images:
        assert widget._elements[name] == "images"
    for name in sdata.labels:
        assert widget._elements[name] == "labels"
    for name in sdata.points:
        assert widget._elements[name] == "points"
    for name in sdata.shapes:
        assert widget._elements[name] == "shapes"


def test_coordinatewidget(make_napari_viewer: Any):
    _ = make_napari_viewer()
    widget = CoordinateSystemWidget(sdata)
    items = [widget.item(x).text() for x in range(widget.count())]
    assert len(items) == len(sdata.coordinate_systems)
    for item in items:
        assert item in sdata.coordinate_systems


def test_sdatawidget_images(make_napari_viewer: Any):
    viewer = make_napari_viewer()
    widget = SdataWidget(viewer, sdata)
    assert len(widget._viewer.layers) == 0
    widget.coordinate_system_widget._select_coord_sys("global")
    widget.elements_widget._onClickChange("global")
    widget._onClick(list(sdata.images.keys())[0])
    assert len(widget._viewer.layers) == 1
    assert isinstance(widget._viewer.layers[0], Image)
    assert widget._viewer.layers[0].name == list(sdata.images.keys())[0]
    assert isinstance(widget._viewer.layers[0].metadata.get("adata"), AnnData)
    assert widget._viewer.layers[0].metadata.get("adata").shape == (0, 0)
    sdata.images["image"] = to_multiscale(sdata.images["blobs_image"], [2, 4])
    widget.elements_widget._onClickChange("global")
    widget._onClick("image")
    assert len(widget._viewer.layers) == 2
    assert (widget._viewer.layers[0].data == widget._viewer.layers[1].data).all()
    del sdata.images["image"]


def test_sdatawidget_labels(make_napari_viewer: Any):
    viewer = make_napari_viewer()
    widget = SdataWidget(viewer, sdata)
    assert len(widget._viewer.layers) == 0
    widget.coordinate_system_widget._select_coord_sys("global")
    widget.elements_widget._onClickChange("global")
    widget._onClick(list(sdata.labels.keys())[0])
    assert len(widget._viewer.layers) == 1
    assert widget._viewer.layers[0].name == list(sdata.labels.keys())[0]
    assert isinstance(widget._viewer.layers[0], Labels)
    assert isinstance(widget._viewer.layers[0].metadata.get("adata"), AnnData)
    assert (
        widget._viewer.layers[0].metadata.get("adata").n_obs
        == (sdata.table.obs[sdata.table.uns["spatialdata_attrs"]["region_key"]] == list(sdata.labels.keys())[0]).sum()
    )
    assert widget._viewer.layers[0].metadata.get("labels_key") is not None


def test_sdatawidget_points(caplog, make_napari_viewer: Any):
    viewer = make_napari_viewer()
    widget = SdataWidget(viewer, sdata)
    assert len(widget._viewer.layers) == 0
    widget.coordinate_system_widget._select_coord_sys("global")
    widget.elements_widget._onClickChange("global")
    widget._onClick(list(sdata.points.keys())[0])
    assert len(widget._viewer.layers) == 1
    assert widget._viewer.layers[0].name == list(sdata.points.keys())[0]
    assert isinstance(widget._viewer.layers[0], Points)
    assert isinstance(widget._viewer.layers[0].metadata.get("adata"), AnnData)
    assert widget._viewer.layers[0].metadata.get("adata").n_obs == len(sdata.points["blobs_points"]["x"])
    assert len(widget._viewer.layers[0].metadata.get("adata").obs.keys()) == sdata.points["blobs_points"].shape[1]
    sdata.points["many_points"] = from_dask_array(randint(0, 10, [200000, 2], dtype=int64), columns=["x", "y"])
    set_transformation(sdata.points["many_points"], {"global": Identity()}, set_all=True)
    widget._add_points("many_points")
    with caplog.at_level(logging.INFO):
        assert (
            "Subsampling points because the number of points exceeds the currently supported 100 000."
            in caplog.records[0].message
        )
    assert widget._viewer.layers[1].metadata.get("adata").n_obs == 100000
    del sdata.points["many_points"]
