from typing import Any

from anndata import AnnData
import pytest

from napari_spatialdata._view import QtAdataScatterWidget
from napari_spatialdata._utils import NDArrayA


# make_napari_viewer is a pytest fixture that returns a napari viewer object
@pytest.mark.parametrize("widget", [QtAdataScatterWidget])
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


# @pytest.mark.parametrize("widget", [QtAdataScatterWidget])
# def test_creating_widget_with_no_adata(make_napari_viewer: Any, widget: Any) -> None:
#     # make viewer and add an image layer using our fixture
#     viewer = make_napari_viewer()

#     # create our widget, passing in the viewer
#     with pytest.raises(NotImplementedError, match=r"`AnnData` not found."):
#         _ = widget(viewer)


# @pytest.mark.parametrize("widget", [QtAdataScatterWidget])
# def test_model(
#     make_napari_viewer: Any,
#     widget: Any,
#     labels: NDArrayA,
#     adata_labels: AnnData,
# ) -> None:
#     # make viewer and add an image layer using our fixture
#     viewer = make_napari_viewer()

#     viewer.add_labels(
#         labels,
#         name="labels",
#         metadata={"adata": adata_labels, "library_id": "labels", "labels_key": "cell_id"},
#     )

#     widget = widget(viewer)
#     layer = viewer.layers.selection.active
#     widget._layer_selection_widget(layer)
#     assert isinstance(widget.model, ImageModel)
#     assert widget.model.library_id == "labels"
#     assert widget.model.adata is adata_labels
#     assert widget.model.coordinates.shape[0] == adata_labels.shape[0]
# assert widget.model.coordinates.ndim == 2
# assert widget.model.labels_key == "cell_id"
