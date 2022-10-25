from typing import Any

from napari_spatialdata._model import ImageModel
from napari_spatialdata._scatterwidgets import MatplotlibWidget


def test_matplotlib_widget(make_napari_viewer: Any):
    # Smoke test adding a matplotlib widget
    viewer = make_napari_viewer()

    import numpy as np

    viewer.add_image(np.random.random((100, 100)))
    viewer.add_labels(np.random.randint(0, 5, (100, 100)))
    MatplotlibWidget(viewer, ImageModel)


# TODO: Add more tests (tests for napari-matplotlib seems like a good place to start)
# https://github.com/matplotlib/napari-matplotlib/blob/main/src/napari_matplotlib/tests/test_scatter.py
