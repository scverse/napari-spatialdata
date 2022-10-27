from typing import Any, Union

from anndata import AnnData
from napari.layers import Image, Labels, Points
import numpy as np
import pytest

from napari_spatialdata._view import QtAdataViewWidget, QtAdataScatterWidget
from napari_spatialdata._model import ImageModel
from napari_spatialdata._utils import NDArrayA


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
        metadata={"adata": adata_shapes, "library_id": "shapes"},
    )

    # create our widget, passing in the viewer
    _ = widget(viewer)


@pytest.mark.parametrize("widget", [QtAdataViewWidget, QtAdataScatterWidget])
def test_creating_widget_with_no_adata(make_napari_viewer: Any, widget: Any) -> None:
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()

    # create our widget, passing in the viewer
    with pytest.raises(NotImplementedError, match=r"`AnnData` not found."):
        _ = widget(viewer)


@pytest.mark.parametrize("widget", [QtAdataViewWidget, QtAdataScatterWidget])
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
        metadata={"adata": adata_labels, "library_id": "labels", "labels_key": "cell_id"},
    )

    widget = widget(viewer)
    layer = viewer.layers.selection.active
    widget._layer_selection_widget(layer)
    assert isinstance(widget.model, ImageModel)
    assert widget.model.library_id == "labels"
    assert widget.model.adata is adata_labels
    assert widget.model.coordinates.shape[0] == adata_labels.shape[0]
    assert widget.model.coordinates.ndim == 2
    assert widget.model.labels_key == "cell_id"


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
        metadata={"adata": adata_labels, "library_id": "labels", "labels_key": "cell_id"},
    )

    widget = widget(viewer)
    layer = viewer.layers.selection.active
    widget._layer_selection_widget(layer)
    assert isinstance(widget.model, ImageModel)
    assert isinstance(widget.model.layer, Labels)
    assert widget.model.library_id == "labels"

    # select observations
    widget.obs_widget._onAction(items=[obs_item])
    assert isinstance(viewer.layers.selection.active, Labels)
    assert viewer.layers.selection.active.name == f"{obs_item}:{layer_name}"

    # select genes
    widget.var_widget._onAction(items=[var_item])
    assert isinstance(viewer.layers.selection.active, Labels)
    assert viewer.layers.selection.active.name == f"{var_item}:X:{layer_name}"
    assert "perc" in viewer.layers.selection.active.metadata.keys()
    assert "minmax" in viewer.layers.selection.active.metadata.keys()

    layer_name = "image"
    viewer.add_image(
        image,
        rgb=True,
        name=layer_name,
        metadata={"adata": adata_shapes, "library_id": "shapes"},
    )

    layer = viewer.layers.selection.active
    widget._layer_selection_widget(layer)

    assert isinstance(widget.model, ImageModel)
    assert isinstance(widget.model.layer, Image)
    assert widget.model.library_id == "shapes"

    # select observations
    widget.obs_widget._onAction(items=[obs_item])
    assert isinstance(viewer.layers.selection.active, Points)
    assert viewer.layers.selection.active.name == f"{obs_item}:{layer_name}"

    # select genes
    widget.var_widget._onAction(items=[var_item])
    assert isinstance(viewer.layers.selection.active, Points)
    assert viewer.layers.selection.active.name == f"{var_item}:X:{layer_name}"
    assert "perc" in viewer.layers.selection.active.metadata.keys()
    assert "minmax" in viewer.layers.selection.active.metadata.keys()

    # check adata layers
    assert len(widget._get_adata_layer()) == 1
    assert widget._get_adata_layer()[0] is None


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
        metadata={"adata": adata_labels, "library_id": "labels", "labels_key": "cell_id"},
    )

    widget = widget(viewer)
    layer = viewer.layers.selection.active
    widget._layer_selection_widget(layer)
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
        metadata={"adata": adata_labels, "library_id": "labels", "labels_key": "cell_id"},
    )

    widget = widget(viewer)
    layer = viewer.layers.selection.active
    widget._layer_selection_widget(layer)

    widget.x_widget.widget.setAttribute(attr)
    widget.x_widget.widget._onAction(items=[item])

    widget.color_widget.widget.setAttribute(attr)
    widget.color_widget.widget._onAction(items=[item])

    assert widget.x_widget.widget.data.dtype.name == "category"
    assert widget.color_widget.widget.data.dtype.name != "category"
    assert isinstance(widget.color_widget.widget.data, np.ndarray)

    with pytest.raises(ValueError) as err:
        widget.y_widget.widget.setAttribute("nothing")
    assert "nothing is not a valid adata field." in str(err.value)
