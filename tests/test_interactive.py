from napari_spatialdata._interactive import Interactive
from spatialdata import SpatialData

from tests.conftest import PlotTester, PlotTesterMeta


class TestImages(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_can_add_element_image(self, sdata_blobs: SpatialData):
        i = Interactive(sdata=sdata_blobs, headless=True)
        i.add_element(coordinate_system_name="global", element="blobs_image")

    def test_plot_can_add_element_label(self, sdata_blobs: SpatialData):
        i = Interactive(sdata=sdata_blobs, headless=True)
        i.add_element(coordinate_system_name="global", element="blobs_labels")

    def test_plot_can_add_element_multiple(self, sdata_blobs: SpatialData):
        i = Interactive(sdata=sdata_blobs, headless=True)
        i.add_element(coordinate_system_name="global", element="blobs_image")
        i.add_element(coordinate_system_name="global", element="blobs_labels")
        i.add_element(coordinate_system_name="global", element="blobs_circles")
