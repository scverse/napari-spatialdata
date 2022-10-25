from typing import Any

from anndata import AnnData
from napari.layers import Image, Labels, Points
import pytest

from napari_spatialdata._view import QtAdataViewWidget
from napari_spatialdata._model import ImageModel
from napari_spatialdata._utils import NDArrayA


# make_napari_viewer is a pytest fixture that returns a napari viewer object
@pytest.mark.parametrize("widget", [QtAdataViewWidget])
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


@pytest.mark.parametrize("widget", [QtAdataViewWidget])
def test_creating_widget_with_no_adata(make_napari_viewer: Any, widget: Any) -> None:
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()

    # create our widget, passing in the viewer
    with pytest.raises(NotImplementedError, match=r"`AnnData` not found."):
        _ = widget(viewer)


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
