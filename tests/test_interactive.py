from napari_spatialdata._interactive import Interactive
from spatialdata import SpatialData

from tests.conftest import PlotTester, PlotTesterMeta


class TestImages(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_can_add_element_image(self, sdata_blobs: SpatialData):
        i = Interactive(sdata=sdata_blobs, headless=True)
        i.add_element(element="blobs_image", element_coordinate_system="global")

    def test_plot_can_add_element_label(self, sdata_blobs: SpatialData):
        i = Interactive(sdata=sdata_blobs, headless=True)
        i.add_element(element="blobs_labels", element_coordinate_system="global")

    def test_plot_can_add_element_multiple(self, sdata_blobs: SpatialData):
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
        i._viewer.close()


def test_plot_can_add_element_switch_cs(sdata_blobs: SpatialData):
    i = Interactive(sdata=sdata_blobs, headless=True)
    i.add_element(element="blobs_image", element_coordinate_system="global", view_element_system=True)
    assert i._sdata_widget.coordinate_system_widget._system == "global"
    assert i._viewer.layers[-1].visible
    i._viewer.close()
