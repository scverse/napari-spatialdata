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


# @pytest.mark.qt()
# class TestNapari(PlotTester, metaclass=PlotTesterMeta):

#     def test_add_layer(self, labels: NDArrayA, image: NDArrayA, adata_labels: AnnData, adata_shapes: AnnData):


#     def test_add_same_layer(self, qtbot, adata: AnnData, napari_cont: Container, capsys):
#         from napari.layers import Points

#         s.logfile = sys.stderr
#         s.verbosity = 4

#         viewer = Interactive(napari_cont, adata)
#         cnt = viewer._controller

#         data = np.random.normal(size=adata.n_obs)
#         cnt.add_points(data, layer_name="layer1")
#         cnt.add_points(np.random.normal(size=adata.n_obs), layer_name="layer1")

#         err = capsys.readouterr().err

#         assert "Layer `layer1` is already loaded" in err
#         assert len(viewer._controller.view.layers) == 2
#         assert viewer._controller.view.layernames == {"V1_Adult_Mouse_Brain", "layer1"}
#         assert isinstance(viewer._controller.view.layers["layer1"], Points)
#         np.testing.assert_array_equal(viewer._controller.view.layers["layer1"].metadata["data"], data)
