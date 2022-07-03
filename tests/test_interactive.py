import pytest

from napari_spatialdata._view import QtAdataViewWidget


# make_napari_viewer is a pytest fixture that returns a napari viewer object
@pytest.mark.parametrize("widget", [QtAdataViewWidget])
def test_creating_widget_with_data(make_napari_viewer, widget, image, adata_shapes):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_image(
        image,
        rgb=True,
        name="image",
        # metadata={"ImageModel": ImageModel(adata1, "point8")}
        metadata={"adata": adata_shapes, "library_id": "shapes"},
    )

    # create our widget, passing in the viewer
    _ = widget(viewer)


@pytest.mark.parametrize("widget", [QtAdataViewWidget])
def test_creating_widget_with_no_adata(make_napari_viewer, widget):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()

    # create our widget, passing in the viewer
    with pytest.raises(NotImplementedError, match=r"`AnnData` not found."):
        _ = widget(viewer)
