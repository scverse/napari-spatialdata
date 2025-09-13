import platform
import sys
from collections.abc import Callable
from importlib.metadata import version

import pytest
from napari import Viewer
from packaging.version import parse as parse_version
from pytestqt.qtbot import QtBot
from spatialdata import SpatialData
from spatialdata.models import Image2DModel

from napari_spatialdata._interactive import Interactive
from tests.conftest import PlotTester, PlotTesterMeta, OFFSCREEN

ARM_PROBLEM = (
    parse_version(version("numpy")) < parse_version("2") and sys.platform == "darwin" and platform.machine() == "arm64"
)


@pytest.mark.skipif(OFFSCREEN, reason="Not running in offscreen mode")
@pytest.mark.usefixtures("mock_app_model")
class TestImages(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_can_add_element_image(self, sdata_blobs: SpatialData):
        blobs_image = Image2DModel.parse(sdata_blobs["blobs_image"], c_coords=("r", "g", "b"))
        sdata_blobs["blobs_image"] = blobs_image
        i = Interactive(sdata=sdata_blobs, headless=True)
        i.add_element(element="blobs_image", element_coordinate_system="global")

    def test_plot_can_add_element_label(self, sdata_blobs: SpatialData):
        i = Interactive(sdata=sdata_blobs, headless=True)
        i.add_element(element="blobs_labels", element_coordinate_system="global")

    def test_plot_can_add_element_multiple(self, sdata_blobs: SpatialData):
        blobs_image = Image2DModel.parse(sdata_blobs["blobs_image"], c_coords=("r", "g", "b"))
        sdata_blobs["blobs_image"] = blobs_image
        i = Interactive(sdata=sdata_blobs, headless=True)
        i.add_element(element="blobs_image", element_coordinate_system="global")
        i.add_element(element="blobs_labels", element_coordinate_system="global")
        i.add_element(element="blobs_circles", element_coordinate_system="global")
        assert not i._sdata_widget.coordinate_system_widget._system
        assert not i._sdata_widget.elements_widget._elements
        for layer in i._viewer.layers:
            assert layer.visible

    def test_switch_coordinate_system(self, sdata_blobs: SpatialData):
        i = Interactive(sdata=sdata_blobs, headless=True)
        assert not i._sdata_widget.coordinate_system_widget._system
        assert not i._sdata_widget.elements_widget._elements
        i.switch_coordinate_system("global")
        assert i._sdata_widget.coordinate_system_widget._system == "global"
        assert i._sdata_widget.elements_widget._elements
        Viewer.close_all()


@pytest.mark.skipif(OFFSCREEN, reason="Not running in offscreen mode")
@pytest.mark.usefixtures("mock_app_model")
def test_plot_can_add_element_switch_cs(sdata_blobs: SpatialData):
    i = Interactive(sdata=sdata_blobs, headless=True)
    i.add_element(element="blobs_image", element_coordinate_system="global", view_element_system=True)
    assert i._sdata_widget.coordinate_system_widget._system == "global"
    assert i._viewer.layers[-1].visible
    Viewer.close_all()


@pytest.mark.skipif(OFFSCREEN, reason="Not running in offscreen mode")
@pytest.mark.usefixtures("mock_app_model")
class TestInteractive(PlotTester, metaclass=PlotTesterMeta):
    def test_get_layer_existing(self, sdata_blobs: SpatialData):
        i = Interactive(sdata=sdata_blobs, headless=True)
        i.add_element(element="blobs_image", element_coordinate_system="global")
        layer = i.get_layer("blobs_image")
        assert layer is not None, "Expected to retrieve the blobs_image layer, but got None"
        assert layer.name == "blobs_image", f"Expected layer name 'blobs_image', got {layer.name}"
        Viewer.close_all()

    def test_get_layer_non_existing(self, sdata_blobs: SpatialData):
        i = Interactive(sdata=sdata_blobs, headless=True)
        layer = i.get_layer("non_existing_layer")
        assert layer is None, "Expected None for a non-existing layer, but got a layer"
        Viewer.close_all()

    def test_add_text_to_polygons(self, sdata_blobs: SpatialData):
        i = Interactive(sdata=sdata_blobs, headless=True)
        i.add_element(element="blobs_polygons", element_coordinate_system="global")

        # Mock polygon layer with some polygon data
        text_annotations = ["Label 1", "Label 2", "Label 3", "Label 4", "Label 5"]
        polygon_layer = i.get_layer("blobs_polygons")

        # Verify that text is added
        i.add_text_to_polygons(layer_name="blobs_polygons", text_annotations=text_annotations)
        assert polygon_layer.text is not None, "Text annotations were not added to the polygon layer"
        Viewer.close_all()


@pytest.mark.skipif(OFFSCREEN, reason="Not running in offscreen mode")
@pytest.mark.skipif(ARM_PROBLEM, reason="Test will segfault on ARM with numpy < 2")
@pytest.mark.use_thread_loader
def test_load_data_in_thread(make_napari_viewer: Callable[[], Viewer], sdata_blobs: SpatialData, qtbot: QtBot) -> None:
    viewer = make_napari_viewer()
    i = Interactive(sdata=sdata_blobs, headless=True)
    with qtbot.waitSignal(i._sdata_widget.worker_thread.finished):
        i.add_element(element="blobs_image", element_coordinate_system="global")
    assert "blobs_image" in viewer.layers
    Viewer.close_all()
