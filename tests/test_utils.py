import logging
from typing import Any, List

import numpy as np
import pytest
from anndata import AnnData
from napari_spatialdata._utils import (
    _get_categorical,
    _get_transform,
    _min_max_norm,
    _points_inside_triangles,
    _position_cluster_labels,
    _set_palette,
)
from spatialdata.datasets import blobs


def test_get_categorical(adata_labels: AnnData):
    assert _get_categorical(adata_labels, key="categorical").shape == (adata_labels.n_obs, 3)


def test_set_palette(adata_labels: AnnData):
    assert np.array_equal(
        _get_categorical(adata_labels, key="categorical"),
        _get_categorical(adata_labels, key="categorical", colordict=_set_palette(adata_labels, key="categorical")),
    )


def test_value_error(adata_labels: AnnData):
    col_dict = _set_palette(adata_labels, key="categorical")
    col_dict[1] = "non_existing_color"
    with pytest.raises(ValueError) as err:
        _get_categorical(adata_labels, key="categorical", colordict=col_dict)
    assert "`non_existing_color` is not an acceptable color." in str(err.value)
    col_dict[27] = col_dict.pop(1)
    with pytest.raises(ValueError) as err:
        _get_categorical(adata_labels, key="categorical", colordict=col_dict)
    assert "The key `27` in the given dictionary is not an existing category in anndata[`categorical`]." in str(
        err.value
    )


def test_position_cluster_labels(adata_labels: AnnData):
    from napari_spatialdata._model import ImageModel

    model = ImageModel()
    model.coordinates = np.insert(adata_labels.obsm["spatial"], 0, values=0, axis=1).copy()
    clusters = adata_labels.obs["categorical"]

    positions = _position_cluster_labels(model.coordinates, clusters)
    assert isinstance(positions["clusters"], np.ndarray)
    assert np.unique(positions["clusters"].nonzero()).shape == adata_labels.obs["categorical"].cat.categories.shape


@pytest.mark.parametrize("tri_coord", [[[0, 10, 20], [30, 40, 50]]])
def test_points_inside_triangles(adata_shapes: AnnData, tri_coord: List[List[int]]):
    from napari_spatialdata._model import ImageModel

    coord1, coord2 = tri_coord

    model = ImageModel()
    model.coordinates = np.insert(adata_shapes.obsm["spatial"], 0, values=0, axis=1).copy()

    triangles = np.array(
        [
            model.coordinates[coord1, ...][:, 1:],
            model.coordinates[coord2, ...][:, 1:],
        ]
    )

    out = _points_inside_triangles(model.coordinates[:, 1:], triangles)

    assert out.shape[0] == model.coordinates.shape[0]
    assert out.any()


@pytest.mark.parametrize("vec", [np.array([0, 0, 2]), np.array([1, 1, 0])])
def test_min_max_norm(vec: np.ndarray) -> None:
    out = _min_max_norm(vec)

    assert out.shape == vec.shape
    assert (out.min(), out.max()) == (0, 1)


def test_logger(caplog, adata_labels: AnnData, make_napari_viewer: Any):
    from napari_spatialdata._model import ImageModel
    from napari_spatialdata._scatterwidgets import MatplotlibWidget

    viewer = make_napari_viewer()
    model = ImageModel()

    m = MatplotlibWidget(viewer, model)
    m._onClick(np.ones(10), np.ones(10), np.ones(10), "X", "Y", "Color")

    with caplog.at_level(logging.INFO):
        assert "Plotting" in caplog.records[0].message


def test_get_transform():
    sdata = blobs()
    assert (_get_transform(sdata.images["blobs_image"]) == np.identity(3)).all()
