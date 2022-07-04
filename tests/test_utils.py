from typing import List

from anndata import AnnData
import numpy as np
import pytest

from napari_spatialdata._utils import (
    _min_max_norm,
    _get_categorical,
    _points_inside_triangles,
    _position_cluster_labels,
)


# test _get_categorical
def test_get_categorical(adata_labels: AnnData):

    assert _get_categorical(adata_labels, key="categorical").shape == (adata_labels.n_obs, 3)


def test_position_cluster_labels(adata_labels: AnnData):
    from napari_spatialdata._model import ImageModel

    model = ImageModel()
    model.coordinates = np.insert(adata_labels.obsm["spatial"], 0, values=0, axis=1).copy()
    clusters = adata_labels.obs["categorical"]
    colors = _get_categorical(adata_labels, key="categorical")

    positions = _position_cluster_labels(model.coordinates, clusters, colors)

    assert isinstance(positions["clusters"], np.ndarray)
    assert isinstance(positions["colors"], np.ndarray)
    assert positions["clusters"].shape == positions["colors"].shape
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
