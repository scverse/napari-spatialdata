import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from shapely.geometry import Polygon

from napari_spatialdata._model import DataModel
from napari_spatialdata._scatterwidgets import PlotWidget

pytestmark = pytest.mark.usefixtures("mock_app_model")


DATA_LEN = 100


def normalize_coordinates(coords):
    return [tuple(map(float, coord)) for coord in coords]


def coordinates_are_equal(coords1, coords2, tol=1e-6):
    norm_coords1 = normalize_coordinates(coords1)
    norm_coords2 = normalize_coordinates(coords2)

    if len(norm_coords1) != len(norm_coords2):
        return False

    return all(np.allclose(c1, c2, atol=tol) for c1, c2 in zip(norm_coords1, norm_coords2, strict=True))


@pytest.fixture
def plot_widget(qtbot, request):
    """Fixture for creating a PlotWidget instance."""
    model = DataModel()
    widget = PlotWidget(None, model)
    qtbot.addWidget(widget)

    def cleanup():
        widget.close()
        QtWidgets.QApplication.processEvents()  # flush events

    request.addfinalizer(cleanup)

    return widget


def test_initialization(plot_widget):
    """Test initialization of PlotWidget."""
    assert plot_widget.is_widget is True
    assert plot_widget.scatter_plot is not None
    assert plot_widget.hovered_point is not None
    assert plot_widget.auto_range_button is not None
    assert plot_widget.drawing_mode_button is not None
    assert plot_widget.rectangle_mode_button is not None


def test_plot_data(plot_widget, prepare_continuous_test_data):
    """Test plotting data."""
    plot_widget._onClick(*prepare_continuous_test_data)
    plot_widget.plot()

    assert plot_widget.x_data is not None
    assert plot_widget.y_data is not None
    assert len(plot_widget.scatter.xData) == DATA_LEN
    assert len(plot_widget.scatter.yData) == DATA_LEN


def test_plot_data_cont(plot_widget, prepare_continuous_test_data):
    """Test building lut widget."""
    plot_widget._onClick(*prepare_continuous_test_data)

    assert plot_widget.lut is not None
    assert plot_widget.discrete_color_widget is None


def test_plot_data_discrete(plot_widget, prepare_discrete_test_data):
    """Test building discrete colors widget."""
    plot_widget._onClick(*prepare_discrete_test_data)

    assert plot_widget.scatter is not None
    assert plot_widget.lut is None
    assert plot_widget.discrete_color_widget is not None


def test_plot_data_widget_change(plot_widget, prepare_continuous_test_data, prepare_discrete_test_data):
    """Test building color widgets upon changing data type."""
    plot_widget._onClick(*prepare_discrete_test_data)

    assert plot_widget.discrete_color_widget is not None
    assert plot_widget.lut is None

    # change from discrete to continuous
    plot_widget._onClick(*prepare_continuous_test_data)
    assert plot_widget.lut is not None
    assert plot_widget.discrete_color_widget is None

    # change from continuous to discrete
    plot_widget._onClick(*prepare_discrete_test_data)
    assert plot_widget.discrete_color_widget is not None
    assert plot_widget.lut is None


def test_plot_no_data(plot_widget):
    """Test plotting no data."""
    plot_widget._onClick(None, None, None, "None: None", "None: None", "None: None")
    plot_widget.plot()

    assert plot_widget.x_label is None
    assert plot_widget.scatter is None


def test_plot_pseudo_histogram(plot_widget, prepare_continuous_test_data):
    """Test plotting no data."""
    x_data, y_data, color_data, x_label, y_label, color_label = prepare_continuous_test_data
    plot_widget._onClick(None, y_data, color_data, "None: None", y_label, color_label)
    plot_widget.plot()

    assert plot_widget.x_label == "Count"
    assert plot_widget.scatter is not None

    plot_widget._onClick(x_data, None, color_data, x_label, "None: None", color_label)
    plot_widget.plot()

    assert plot_widget.y_label == "Count"
    assert plot_widget.scatter is not None


def test_hover_highlight_cont(plot_widget, prepare_continuous_test_data):
    """Test hover highlight functionality."""
    x_data, y_data, color_data, x_label, y_label, color_label = prepare_continuous_test_data
    plot_widget._onClick(x_data, y_data, color_data, x_label, y_label, color_label)
    plot_widget.plot()

    plot_widget.update_hover_highlight(x_data["vec"][0], y_data["vec"][0])
    assert plot_widget.hovered_point.data[0][0] == x_data["vec"][0]
    assert plot_widget.hovered_point.data[0][1] == y_data["vec"][0]

    plot_widget.update_hover_highlight(-1, -1)
    assert plot_widget.data_point_label.text() == "Value: N/A"
    assert plot_widget.hovered_point.data.size == 0


def test_clear_hover_highlight(plot_widget, prepare_discrete_test_data):
    """Test clearing of hover highlight."""
    x_data, y_data, color_data, x_label, y_label, color_label = prepare_discrete_test_data
    plot_widget._onClick(x_data, y_data, color_data, x_label, y_label, color_label)
    plot_widget.plot()

    plot_widget.update_hover_highlight(x_data["vec"][0], y_data["vec"][0])
    assert plot_widget.hovered_point.data[0][0] == x_data["vec"][0]
    assert plot_widget.hovered_point.data[0][1] == y_data["vec"][0]

    plot_widget.clear_hover_highlight()
    assert plot_widget.hovered_point.data.size == 0


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


def test_roi_from_mouse_events(qtbot, plot_widget):
    """Test mouse press and release events for drawing ROIs."""
    plot_widget.toggle_drawing_mode()
    assert plot_widget.drawing is True

    plot_widget.scatter_plot.setRange(xRange=[0, 100], yRange=[0, 100])

    # point coordinates in plot coordinates
    point1 = [10, 10]
    point2 = [70, 70]
    point3 = [10, 80]

    # Convert plot coordinates to viewport coordinates
    viewport_pos1 = plot_coords_to_viewport(plot_widget, *point1)
    viewport_pos2 = plot_coords_to_viewport(plot_widget, *point2)
    viewport_pos3 = plot_coords_to_viewport(plot_widget, *point3)

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

    expected_roi = pg.PolyLineROI([point1, point2, point3], closed=True)
    assert np.allclose(plot_widget.roi_list[0].saveState()["points"], expected_roi.saveState()["points"], atol=2 * 1e-1)

    plot_widget.toggle_rectangle_mode()
    assert plot_widget.rectangle is True

    qtbot.mousePress(view_widget, QtCore.Qt.LeftButton, pos=viewport_pos1)
    send_mouse_event(QtCore.QEvent.MouseMove, view_widget, viewport_pos2)
    QtWidgets.QApplication.processEvents()
    qtbot.mouseRelease(view_widget, QtCore.Qt.LeftButton, pos=viewport_pos2)

    assert len(plot_widget.roi_list) == 2


def test_roi_to_polygon(plot_widget):
    """Test conversion of ROI to Polygon."""
    vertices = [[0, 0], [0, 1], [1, 1], [1, 0]]
    roi = pg.PolyLineROI(vertices, closed=True)
    plot_widget.roi_list = [roi]
    polygon_list = plot_widget.rois_to_polygons()

    assert isinstance(polygon_list[0], Polygon)
    assert coordinates_are_equal(list(polygon_list[0].exterior.coords)[:-1], vertices)


def test_roi_to_polygon_rect(plot_widget):
    """Test conversion of ROI to Polygon."""
    vertices = [[0, 0], [0, 1], [1, 1], [1, 0]]
    roi = pg.RectROI([0, 0], [1, 1])
    plot_widget.roi_list = [roi]
    polygon_list = plot_widget.rois_to_polygons()

    assert isinstance(polygon_list[0], Polygon)

    polygon_vertices = list(polygon_list[0].exterior.coords)[:-1]
    assert coordinates_are_equal(sorted(polygon_vertices), sorted(vertices))


def test_coord_change_remove_roi(plot_widget, prepare_discrete_test_data):
    """Test changing coordinates and removing ROI."""
    roi = pg.RectROI([0, 0], [1, 1])
    plot_widget.roi_list = [roi]

    assert len(plot_widget.roi_list) == 1

    plot_widget._onClick(*prepare_discrete_test_data)
    assert len(plot_widget.roi_list) == 0


def test_add_roi_to_plot(qtbot, plot_widget, prepare_discrete_test_data):
    """Test changing coordinates and removing ROI."""
    roi = pg.RectROI([0, 0], [1, 1])
    plot_widget._onClick(*prepare_discrete_test_data)
    plot_widget.roi_list = [roi]
    plot_widget.plot()

    assert len(plot_widget.roi_list) == 1

    rois = [item for item in plot_widget.scatter_plot.items if isinstance(item, pg.RectROI)]

    assert len(rois) == 1, "There should be exactly one ROI in the scatter_plot"


# sometimes this test fails due to a faulty clean-up, for example in the last failing
# commit here: https://github.com/scverse/napari-spatialdata/pull/360
def test_remove_roi_double_click(qtbot, plot_widget, prepare_discrete_test_data):
    """Test changing coordinates and removing ROI."""
    roi = pg.RectROI([0, 0], [1, 1])
    plot_widget._onClick(*prepare_discrete_test_data)
    plot_widget.roi_list = [roi]
    plot_widget.plot()

    assert len(plot_widget.roi_list) == 1

    # Simulate double click
    view_widget = plot_widget.scatter_plot.scene().views()[0].viewport()

    point1 = [0.5, 0.5]
    viewport_pos1 = plot_coords_to_viewport(plot_widget, *point1)

    qtbot.mouseDClick(view_widget, QtCore.Qt.LeftButton, pos=viewport_pos1)
    qtbot.wait(50)

    assert len(plot_widget.roi_list) == 0


def test_remove_hovered_roi(qtbot, plot_widget, prepare_discrete_test_data):
    """Test changing coordinates and removing ROI."""
    roi = pg.RectROI([0, 0], [1, 1])
    plot_widget.roi_list = [roi]
    plot_widget.plot()

    assert len(plot_widget.roi_list) == 1

    # Simulate hovering over the ROI
    roi.mouseHovering = True

    plot_widget.remove_hovered_roi()

    assert len(plot_widget.roi_list) == 0


def test_remove_all_rois(qtbot, plot_widget, prepare_discrete_test_data):
    """Test changing coordinates and removing ROI."""
    roi1 = pg.RectROI([0, 0], [0.5, 0.5])
    roi2 = pg.RectROI([0.6, 0.6], [0.7, 0.7])
    roi3 = pg.RectROI([0.8, 0.8], [0.9, 0.9])
    plot_widget._onClick(*prepare_discrete_test_data)
    plot_widget.roi_list = [roi1, roi2, roi3]
    plot_widget.plot()

    assert len(plot_widget.roi_list) == 3

    plot_widget.remove_all_rois()

    assert len(plot_widget.roi_list) == 0


def test_keyboard_bindings(qtbot, plot_widget, prepare_discrete_test_data):
    """Test removing rois with keyboard shortcuts."""
    roi1 = pg.RectROI([0, 0], [0.5, 0.5])
    roi2 = pg.RectROI([0.6, 0.6], [0.7, 0.7])
    roi3 = pg.RectROI([0.8, 0.8], [0.9, 0.9])
    plot_widget._onClick(*prepare_discrete_test_data)
    plot_widget.roi_list = [roi1, roi2, roi3]
    plot_widget.plot()

    assert len(plot_widget.roi_list) == 3

    # Mock test - no deletion expected with any Shift + not d key
    event = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_E, QtCore.Qt.ShiftModifier)
    QtWidgets.QApplication.sendEvent(plot_widget, event)
    assert len(plot_widget.roi_list) == 3

    # Test d key - delete hovered ROI
    roi2.mouseHovering = True
    event = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_D, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(plot_widget, event)
    assert len(plot_widget.roi_list) == 2

    # Test Shift + d keys - delete all ROIs
    event = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_D, QtCore.Qt.ShiftModifier)
    QtWidgets.QApplication.sendEvent(plot_widget, event)
    assert len(plot_widget.roi_list) == 0


def test_selection_from_roi(plot_widget, prepare_discrete_test_data):
    """Test selection of points from roi."""
    plot_widget._onClick(*prepare_discrete_test_data)
    roi = pg.RectROI([-0.1, -0.1], [1.2, 1.2])
    plot_widget.roi_list = [roi]

    boolean_vector = plot_widget.get_selection()

    assert len(boolean_vector) == DATA_LEN
    assert np.sum(boolean_vector) == DATA_LEN


def test_auto_range_discrete(plot_widget, prepare_discrete_test_data):
    """Test auto range for discrete data."""
    plot_widget._onClick(*prepare_discrete_test_data)

    label = plot_widget.color_data["labels"][0]

    org_color = plot_widget.discrete_color_widget.color_buttons[label].color().getRgbF()

    new_color = (0.5, 0.5, 0.5, 1.0)

    plot_widget.discrete_color_widget.color_buttons[label].setColor(QtGui.QColor.fromRgbF(*new_color))

    assert np.allclose(plot_widget.discrete_color_widget.color_buttons[label].color().getRgbF(), new_color, atol=1e-03)

    # auto range is supposed to trigger a reset of the color
    plot_widget.use_auto_range()

    assert np.allclose(plot_widget.discrete_color_widget.color_buttons[label].color().getRgbF(), org_color, atol=1e-03)


def test_auto_range_continuous(plot_widget, prepare_continuous_test_data):
    """Test auto range for discrete data."""
    plot_widget._onClick(*prepare_continuous_test_data)

    color_min = plot_widget.color_vec.min()
    color_max = plot_widget.color_vec.max()

    assert np.allclose(plot_widget.lut.getLevels(), (color_min, color_max), atol=1e-03)

    new_color_min = color_min + 0.1
    new_color_max = color_max - 0.1
    plot_widget.lut.setLevels(new_color_min, new_color_max)

    assert np.allclose(plot_widget.lut.getLevels(), (new_color_min, new_color_max), atol=1e-03)

    # auto range is supposed to trigger a reset of the color
    plot_widget.use_auto_range()

    assert np.allclose(plot_widget.lut.getLevels(), (color_min, color_max), atol=1e-03)
