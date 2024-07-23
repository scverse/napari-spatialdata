import numpy as np
import pyqtgraph as pg
import pytest
from napari_spatialdata._model import DataModel
from napari_spatialdata._scatterwidgets import PlotWidget
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

RNG = np.random.default_rng(seed=0)
DATA_LEN = 100


def prepare_cont_test_data():
    x_vec = RNG.random(DATA_LEN)
    y_vec = RNG.random(DATA_LEN)
    color_vec = RNG.random(DATA_LEN)

    x_data = {"vec": x_vec}
    y_data = {"vec": y_vec}
    color_data = {"vec": color_vec}

    x_label = "X-axis"
    y_label = "Y-axis"
    color_label = "Color Label"
    return x_data, y_data, color_data, x_label, y_label, color_label


@pytest.fixture
def plot_widget(qtbot):
    """Fixture for creating a PlotWidget instance."""
    model = DataModel()
    widget = PlotWidget(None, model)
    qtbot.addWidget(widget)
    yield widget
    widget.deleteLater()
    qtbot.wait(50)


def test_initialization(plot_widget):
    """Test initialization of PlotWidget."""
    assert plot_widget.is_widget is True
    assert plot_widget.scatter_plot is not None
    assert plot_widget.hovered_point is not None
    assert plot_widget.auto_range_button is not None
    assert plot_widget.drawing_mode_button is not None
    assert plot_widget.rectangle_mode_button is not None


def test_plot_data_cont(plot_widget):
    """Test plotting data."""
    x_data, y_data, color_data, x_label, y_label, color_label = prepare_cont_test_data()
    plot_widget._onClick(x_data, y_data, color_data, x_label, y_label, color_label)
    plot_widget.plot()

    assert plot_widget.x_data is not None
    assert plot_widget.y_data is not None
    assert len(plot_widget.scatter.xData) == DATA_LEN
    assert len(plot_widget.scatter.yData) == DATA_LEN


def test_hover_highlight(qtbot, plot_widget):
    """Test hover highlight functionality."""
    x_data, y_data, color_data, x_label, y_label, color_label = prepare_cont_test_data()
    plot_widget._onClick(x_data, y_data, color_data, x_label, y_label, color_label)
    plot_widget.plot()

    plot_widget.updateHoverHighlight(x_data["vec"][0], y_data["vec"][0])
    assert plot_widget.hovered_point.data[0][0] == x_data["vec"][0]
    assert plot_widget.hovered_point.data[0][1] == y_data["vec"][0]


def test_toggle_drawing_mode(plot_widget):
    """Test toggling of drawing mode."""
    plot_widget.toggle_drawing_mode()
    assert plot_widget.drawing is True
    assert plot_widget.rectangle is False

    plot_widget.toggle_drawing_mode()
    assert plot_widget.drawing is False


def test_toggle_rectangle_mode(plot_widget):
    """Test toggling of rectangle mode."""
    plot_widget.toggle_rectangle_mode()
    assert plot_widget.rectangle is True
    assert plot_widget.drawing is False

    plot_widget.toggle_rectangle_mode()
    assert plot_widget.rectangle is False


def plot_coords_to_viewport(plot_widget, x, y):
    plot_pos = pg.Point(x, y)
    view_pos = plot_widget.scatter_plot.vb.mapViewToScene(plot_pos)
    return plot_widget.scatter_plot.scene().views()[0].mapFromScene(view_pos)


def send_mouse_event(event_type, widget, pos, button=QtCore.Qt.LeftButton):
    """Send a mouse event to a widget."""
    # necessary because of https://bugreports.qt.io/browse/QTBUG-5232
    event = QtGui.QMouseEvent(event_type, pos, button, button, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(widget, event)


def test_mouse_events(qtbot, plot_widget):
    """Test mouse press and release events for drawing ROIs."""
    plot_widget.toggle_drawing_mode()
    assert plot_widget.drawing is True

    plot_widget.scatter_plot.setRange(xRange=[0, 100], yRange=[0, 100])

    # Convert plot coordinates to viewport coordinates
    viewport_pos1 = plot_coords_to_viewport(plot_widget, 10, 10)
    viewport_pos2 = plot_coords_to_viewport(plot_widget, 70, 70)
    viewport_pos3 = plot_coords_to_viewport(plot_widget, 10, 80)

    view_widget = plot_widget.scatter_plot.scene().views()[0].viewport()

    # Simulate mouse press
    qtbot.mousePress(view_widget, QtCore.Qt.LeftButton, pos=viewport_pos1)
    qtbot.wait(50)
    # Simulate mouse move
    send_mouse_event(QtCore.QEvent.MouseMove, view_widget, viewport_pos2)
    QtWidgets.QApplication.processEvents()
    qtbot.wait(50)
    send_mouse_event(QtCore.QEvent.MouseMove, view_widget, viewport_pos3)
    QtWidgets.QApplication.processEvents()
    qtbot.wait(50)
    # Simulate mouse release
    qtbot.mouseRelease(view_widget, QtCore.Qt.LeftButton, pos=viewport_pos3)
    qtbot.wait(50)

    assert len(plot_widget.roi_list) == 1

    plot_widget.toggle_rectangle_mode()
    assert plot_widget.rectangle is True

    qtbot.mousePress(view_widget, QtCore.Qt.LeftButton, pos=viewport_pos1)
    send_mouse_event(QtCore.QEvent.MouseMove, view_widget, viewport_pos2)
    QtWidgets.QApplication.processEvents()
    qtbot.mouseRelease(view_widget, QtCore.Qt.LeftButton, pos=viewport_pos2)

    assert len(plot_widget.roi_list) == 2
