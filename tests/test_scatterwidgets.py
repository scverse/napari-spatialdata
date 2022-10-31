from typing import Any

import numpy as np

from napari_spatialdata._model import ImageModel
from napari_spatialdata._scatterwidgets import MatplotlibWidget


def prepare_test_data():

    x_data = np.random.random((100, 100))
    y_data = np.random.random((100, 100))
    color_data = np.random.random((100, 100))
    x_label = "X-axis"
    y_label = "Y-axis"
    return x_data, y_data, color_data, x_label, y_label


def test_matplotlib_widget(make_napari_viewer: Any):

    # Smoke test adding a matplotlib widget

    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    viewer.add_labels(np.random.randint(0, 5, (100, 100)))

    MatplotlibWidget(viewer, ImageModel)


def test_matplotlib_widget_plot(make_napari_viewer: Any):

    viewer = make_napari_viewer()
    x_data, y_data, color_data, x_label, y_label = prepare_test_data()
    mpl_widget = MatplotlibWidget(viewer, ImageModel)

    mpl_widget._onClick(x_data, y_data, color_data, x_label, y_label)

    assert mpl_widget.axes.get_xlabel() == x_label
    assert mpl_widget.axes.get_ylabel() == y_label
