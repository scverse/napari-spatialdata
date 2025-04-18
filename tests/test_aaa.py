import pytest
from napari.viewer import Viewer

from napari_spatialdata._model import DataModel
from napari_spatialdata._scatterwidgets import PlotWidget


@pytest.fixture
def plot_widget(qtbot):
    """Fixture for creating a PlotWidget instance."""
    model = DataModel()
    widget = PlotWidget(None, model)
    qtbot.addWidget(widget)
    yield widget


def test_initialization(plot_widget):
    """Test initialization of PlotWidget."""
    assert plot_widget is not None


def test_elementwidget(make_napari_viewer):
    _ = make_napari_viewer()
    assert 1 == 1
    Viewer.close_all()
