from typing import Any

import numpy as np
import pandas as pd
from napari_spatialdata._model import ImageModel
from napari_spatialdata._scatterwidgets import MatplotlibWidget


def prepare_test_data():
    x_data = np.random.random((100, 100))
    y_data = np.random.random((100, 100))
    color_data = np.random.random(10000)
    x_label = "X-axis"
    y_label = "Y-axis"
    color_label = "Color Label"
    return x_data, y_data, color_data, x_label, y_label, color_label


def test_matplotlib_widget(make_napari_viewer: Any):
    # Smoke test: adding a matplotlib widget

    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((100, 100)))
    viewer.add_labels(np.random.randint(0, 5, (100, 100)))  # noqa: NPY002

    MatplotlibWidget(viewer, ImageModel)


def test_matplotlib_widget_plot(make_napari_viewer: Any):
    viewer = make_napari_viewer()
    x_data, y_data, color_data, x_label, y_label, color_label = prepare_test_data()
    mpl_widget = MatplotlibWidget(viewer, ImageModel)

    mpl_widget._onClick(x_data, y_data, color_data, x_label, y_label, color_label)

    assert mpl_widget.axes.get_xlabel() == x_label
    assert mpl_widget.axes.get_ylabel() == y_label

    assert mpl_widget.colorbar is not None
    mpl_widget.clear()
    assert mpl_widget.colorbar is None


def test_interactivity_widget(make_napari_viewer: Any):
    viewer = make_napari_viewer()
    x_data, y_data, color_data, x_label, y_label, color_label = prepare_test_data()
    mpl_widget = MatplotlibWidget(viewer, ImageModel)

    mpl_widget._onClick(x_data, y_data, color_data, x_label, y_label, color_label)
    mpl_widget.selector.onselect(np.ones((100, 2)))

    assert mpl_widget.selector.selected_coordinates.size == 0
    assert np.array_equal(
        mpl_widget.selector.exported_data,
        pd.Categorical(mpl_widget.selector.path.contains_points(mpl_widget.selector.xys)),
    )
