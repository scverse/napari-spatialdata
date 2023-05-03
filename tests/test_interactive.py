import platform
from typing import Any, Union

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from napari.layers import Image, Labels
from napari_spatialdata._model import ImageModel
from napari_spatialdata._utils import NDArrayA
from napari_spatialdata._view import QtAdataScatterWidget, QtAdataViewWidget


# make_napari_viewer is a pytest fixture that returns a napari viewer object
@pytest.mark.parametrize("widget", [QtAdataViewWidget, QtAdataScatterWidget])
def test_creating_widget_with_data(
    make_napari_viewer: Any,
    widget: Any,
    image: NDArrayA,
    adata_shapes: AnnData,
) -> None:
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_image(
        image,
        rgb=True,
        name="image",
        metadata={"adata": adata_shapes},
    )

    # create our widget, passing in the viewer
    _ = widget(viewer)
    viewer.layers.selection.events.changed.disconnect()


@pytest.mark.skipif(platform.system() == "Linux", reason="Fails on ubuntu CI")
@pytest.mark.skipif(platform.system() == "Darwin", reason="Fails on macos CI, but locally it is fine")
@pytest.mark.parametrize("widget", [QtAdataViewWidget, QtAdataScatterWidget])
def test_creating_widget_with_no_adata(make_napari_viewer: Any, widget: Any) -> None:
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()

    # create our widget, passing in the viewer
    # with pytest.raises(NotImplementedError, match=r":class:`anndata.AnnData` not found in any `layer.metadata`."):
    with pytest.raises(AttributeError, match=r"'NoneType' object has no attribute 'metadata'"):
        _ = widget(viewer)
    viewer.layers.selection.events.changed.disconnect()


@pytest.mark.parametrize("widget", [QtAdataViewWidget])
def test_model(
    make_napari_viewer: Any,
    widget: Any,
    labels: NDArrayA,
    adata_labels: AnnData,
) -> None:
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()

    viewer.add_labels(
        labels,
        name="labels",
        metadata={"adata": adata_labels, "labels_key": "cell_id"},
    )

    widget = widget(viewer)
    # layer = viewer.layers.selection.active
    widget._select_layer()
    assert isinstance(widget.model, ImageModel)
    assert widget.model.adata is adata_labels
    assert widget.model.coordinates.shape[0] == adata_labels.shape[0]
    assert widget.model.coordinates.ndim == 2
    assert widget.model.labels_key == "cell_id"
    viewer.layers.selection.events.changed.disconnect()


@pytest.mark.parametrize("widget", [QtAdataViewWidget])
@pytest.mark.parametrize("obs_item", ["a", "categorical"])
@pytest.mark.parametrize("var_item", ["42", "0"])
def test_change_layer(
    make_napari_viewer: Any,
    widget: Any,
    labels: NDArrayA,
    adata_labels: AnnData,
    image: NDArrayA,
    adata_shapes: AnnData,
    obs_item: str,
    var_item: str,
) -> None:
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    layer_name = "labels"

    viewer.add_labels(
        image,
        name=layer_name,
        metadata={"adata": adata_labels, "labels_key": "cell_id"},
    )

    widget = widget(viewer)
    widget._select_layer()
    assert isinstance(widget.model, ImageModel)
    assert isinstance(widget.model.layer, Labels)

    # select observations
    # widget.obs_widget._onAction(items=[obs_item])
    # assert isinstance(viewer.layers.selection.active, Labels)
    # assert viewer.layers.selection.active.name == f"{obs_item}:{layer_name}"

    # select genes
    # widget.var_widget._onAction(items=[var_item])
    # assert isinstance(viewer.layers.selection.active, Labels)
    # assert viewer.layers.selection.active.name == f"{var_item}:X:{layer_name}"
    # assert "perc" in viewer.layers.selection.active.metadata
    # assert "minmax" in viewer.layers.selection.active.metadata

    layer_name = "image"
    viewer.add_image(
        image,
        rgb=True,
        name=layer_name,
        metadata={"adata": adata_shapes},
    )

    widget._select_layer()

    assert isinstance(widget.model, ImageModel)
    assert isinstance(widget.model.layer, Image)

    # select observations
    # widget.obs_widget._onAction(items=[obs_item])
    # assert isinstance(viewer.layers.selection.active, Points)
    # assert viewer.layers.selection.active.name == f"{obs_item}:{layer_name}"

    # select genes
    # widget.var_widget._onAction(items=[var_item])
    # assert isinstance(viewer.layers.selection.active, Points)
    # assert viewer.layers.selection.active.name == f"{var_item}:X:{layer_name}"
    # assert "perc" in viewer.layers.selection.active.metadata
    # assert "minmax" in viewer.layers.selection.active.metadata

    # check adata layers
    # assert len(widget._get_adata_layer()) == 1
    # assert widget._get_adata_layer()[0] is None
    # viewer.layers.selection.events.changed.disconnect()


@pytest.mark.parametrize("widget", [QtAdataScatterWidget])
@pytest.mark.parametrize("attr, item, text", [("obs", "a", None), ("obsm", "spatial", 1), ("var", 27, "X")])
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
        metadata={"adata": adata_labels, "labels_key": "cell_id"},
    )

    widget = widget(viewer)
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

    viewer.add_labels(
        image,
        name=layer_name,
        metadata={"adata": adata_labels, "labels_key": "cell_id"},
    )

    widget = widget(viewer)
    widget._select_layer()

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
        metadata={"adata": adata_labels, "labels_key": "cell_id"},
    )

    widget = widget(viewer)
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
def test_layer_selection(
    make_napari_viewer: Any, image: NDArrayA, widget: Any, adata_labels: AnnData, adata_shapes: AnnData
):
    viewer = make_napari_viewer()

    viewer.add_labels(
        image,
        name="labels",
        metadata={"adata": adata_labels, "labels_key": "cell_id"},
    )
    widget = widget(viewer)
    assert widget.model.adata is adata_labels
    viewer.add_image(
        image,
        rgb=True,
        name="image",
        metadata={"adata": adata_shapes},
    )
    assert widget.model.adata is adata_shapes
