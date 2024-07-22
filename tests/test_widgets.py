from typing import Any, Union

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from anndata.tests.helpers import assert_equal
from napari.layers import Image, Labels
from napari.utils.events import EventedList
from napari_spatialdata._model import DataModel
from napari_spatialdata._sdata_widgets import SdataWidget
from napari_spatialdata._view import QtAdataScatterWidget, QtAdataViewWidget
from napari_spatialdata.utils._utils import NDArrayA
from spatialdata import SpatialData


# make_napari_viewer is a pytest fixture that returns a napari viewer object
@pytest.mark.parametrize("widget", [QtAdataViewWidget, QtAdataScatterWidget])
def test_creating_widget_with_data(
    make_napari_viewer: Any,
    widget: Any,
    sdata_blobs: SpatialData,
    adata_shapes: AnnData,
) -> None:
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    image = np.transpose(sdata_blobs["blobs_image"].data, axes=(1, 2, 0))
    viewer.add_image(
        image,
        rgb=True,
        name="image",
        metadata={"sdata": sdata_blobs, "name": "blobs_image", "adata": adata_shapes},
    )
    model = DataModel()
    # create our widget, passing in the viewer
    _ = widget(viewer, model)
    viewer.layers.selection.events.changed.disconnect()


@pytest.mark.parametrize("widget", [QtAdataViewWidget, QtAdataScatterWidget])
def test_creating_widget_with_no_adata(make_napari_viewer: Any, widget: Any) -> None:
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    model = DataModel()
    # create our widget, passing in the viewer
    _ = widget(viewer, model)
    viewer.layers.selection.events.changed.disconnect()


@pytest.mark.parametrize("widget", [QtAdataViewWidget])
def test_model(
    make_napari_viewer: Any,
    widget: Any,
    labels: NDArrayA,
    sdata_blobs: SpatialData,
) -> None:
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()

    viewer.add_labels(
        sdata_blobs["blobs_labels"],
        name="blobs_labels",
        metadata={
            "sdata": sdata_blobs,
            "name": "blobs_labels",
            "adata": sdata_blobs["table"],
            "region_key": sdata_blobs["table"].uns["spatialdata_attrs"]["region_key"],
            "instance_key": sdata_blobs["table"].uns["spatialdata_attrs"]["instance_key"],
            "table_names": ["table"],
        },
    )
    model = DataModel()
    widget = widget(viewer, model)
    # layer = viewer.layers.selection.active
    widget._select_layer()
    assert isinstance(widget.model, DataModel)
    assert_equal(widget.model.adata, sdata_blobs["table"])

    assert widget.model.region_key == "region"
    assert widget.model.instance_key == "instance_id"
    viewer.layers.selection.events.changed.disconnect()


@pytest.mark.parametrize("widget", [QtAdataViewWidget])
def test_change_layer(
    make_napari_viewer: Any,
    widget: Any,
    sdata_blobs: SpatialData,
) -> None:
    table = sdata_blobs["table"].copy()
    table.obs["region"] = "blobs_labels"
    table.uns["spatialdata_attrs"]["region"] = "blobs_labels"
    table.var_names = pd.Index([i + "_second" for i in table.var_names])
    sdata_blobs["second_table"] = table

    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    sdata_widget = SdataWidget(viewer, EventedList([sdata_blobs]))
    viewer.window.add_dock_widget(sdata_widget, name="SpatialData")
    sdata_widget.viewer_model.add_sdata_image(sdata_blobs, "blobs_image", "global", False)

    widget = widget(viewer)
    widget._select_layer()
    assert isinstance(widget.model, DataModel)
    assert isinstance(widget.model.layer, Image)
    assert widget.table_name_widget.currentText() == ""

    sdata_widget.viewer_model.add_sdata_labels(sdata_blobs, "blobs_labels", "global", False)

    widget._select_layer()

    assert isinstance(widget.model, DataModel)
    assert isinstance(widget.model.layer, Labels)
    assert widget.table_name_widget.currentText() == widget.model.layer.metadata["table_names"][0]


# TODO add back ("obs", "a", None) once adata_labels is adjusted.
@pytest.mark.parametrize("widget", [QtAdataScatterWidget])
@pytest.mark.parametrize("attr, item, text", [("obsm", "spatial", 1), ("var", 27, "X")])
def test_scatterlistwidget(
    make_napari_viewer: Any,
    widget: Any,
    adata_labels: AnnData,
    image: NDArrayA,
    attr: str,
    item: str,
    text: Union[str, int, None],
) -> None:
    viewer = make_napari_viewer()
    layer_name = "labels"

    viewer.add_labels(
        image,
        name=layer_name,
        metadata={"adata": adata_labels, "region_key": "cell_id"},
    )
    model = DataModel()
    widget = widget(viewer, model)
    widget._select_layer()
    # change attr

    widget.x_widget.selection_widget.setCurrentText(attr)
    assert widget.x_widget.widget.getAttribute() == attr
    assert widget.x_widget.component_widget.attr == attr
    widget.x_widget.widget.setComponent(text)
    assert widget.x_widget.widget.text == text

    widget.x_widget.widget._onAction(items=[item])
    if attr == "obsm":
        assert np.array_equal(widget.x_widget.widget.data, getattr(adata_labels, attr)[item][:, text])
    elif attr == "obs":
        assert np.array_equal(widget.x_widget.widget.data, getattr(adata_labels, attr)[item])
    else:
        assert np.array_equal(widget.x_widget.widget.data, adata_labels.X[:, item])


# TODO fix adata_labels as this does not annotate element, tests fail
@pytest.mark.xfail
@pytest.mark.parametrize("widget", [QtAdataScatterWidget])
@pytest.mark.parametrize("attr, item", [("obs", "categorical")])
def test_categorical_and_error(
    make_napari_viewer: Any,
    widget: Any,
    adata_labels: AnnData,
    image: NDArrayA,
    attr: str,
    item: str,
) -> None:
    viewer = make_napari_viewer()
    layer_name = "labels"
    widget = widget(viewer)

    viewer.add_labels(
        image,
        name=layer_name,
        metadata={"adata": adata_labels, "region_key": "cell_id"},
    )

    # widget._select_layer()

    widget.x_widget.widget.setAttribute(attr)
    widget.x_widget.widget._onAction(items=[item])

    widget.color_widget.widget.setAttribute(attr)
    widget.color_widget.widget._onAction(items=[item])

    assert widget.x_widget.widget.data.dtype.name == "category"
    assert isinstance(widget.color_widget.widget.data, dict)
    assert isinstance(widget.color_widget.widget.data["vec"], np.ndarray)
    assert isinstance(widget.color_widget.widget.data["palette"], dict)
    assert widget.color_widget.widget.data["cat"].dtype.name == "category"

    with pytest.raises(ValueError) as err:
        widget.y_widget.widget.setAttribute("nothing")
    assert "nothing is not a valid adata field." in str(err.value)


@pytest.mark.parametrize("widget", [QtAdataScatterWidget])
def test_component_widget(
    make_napari_viewer: Any,
    widget: Any,
    adata_labels: AnnData,
    image: NDArrayA,
) -> None:
    viewer = make_napari_viewer()
    layer_name = "labels"

    viewer.add_labels(
        image,
        name=layer_name,
        metadata={"adata": adata_labels, "region_key": "cell_id"},
    )
    model = DataModel()
    widget = widget(viewer, model)
    widget._select_layer()

    widget.x_widget.selection_widget.setCurrentText("obsm")
    for i in range(widget.x_widget.widget.count()):
        widget.x_widget.component_widget._onClickChange(widget.x_widget.widget.item(i))
        if isinstance(adata_labels.obsm[widget.x_widget.widget.item(i).text()], pd.DataFrame):
            assert (
                (
                    widget.x_widget.component_widget.itemText(i)
                    in adata_labels.obsm[widget.x_widget.widget.item(i).text()]
                )
                for i in range(widget.x_widget.component_widget.count())
            )
        else:
            assert (
                widget.x_widget.component_widget.count()
                == np.shape(adata_labels.obsm[widget.x_widget.widget.item(i).text()])[1]
            )

    widget.x_widget.selection_widget.setCurrentText("obs")
    assert widget.x_widget.component_widget.count() == 0

    widget.x_widget.selection_widget.setCurrentText("var")
    assert widget.x_widget.component_widget.count() == len(adata_labels.layers.keys()) + 1
    assert (
        widget.x_widget.component_widget.itemText(i) in [adata_labels.layers.keys(), "X"]
        for i in range(widget.x_widget.component_widget.count())
    )


@pytest.mark.parametrize("widget", [QtAdataViewWidget, QtAdataScatterWidget])
def test_layer_selection(make_napari_viewer: Any, image: NDArrayA, widget: Any, sdata_blobs: SpatialData):
    viewer = make_napari_viewer()
    sdata_widget = SdataWidget(viewer, EventedList([sdata_blobs]))
    sdata_widget.viewer_model.add_sdata_labels(sdata_blobs, "blobs_labels", "global", False)
    viewer.window.add_dock_widget(sdata_widget, name="SpatialData")

    widget = widget(viewer)
    assert_equal(widget.model.adata.copy(), sdata_blobs["table"])

    sdata_widget.viewer_model.add_sdata_image(sdata_blobs, "blobs_image", "global", False)

    # table is annotating blobs labels so there should be no matching rows.
    assert widget.model.adata.n_obs == 0
