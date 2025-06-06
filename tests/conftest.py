from __future__ import annotations

import random
import string
from abc import ABC, ABCMeta
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import napari
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from loguru import logger
from matplotlib.testing.compare import compare_images
from scipy import ndimage as ndi
from skimage import data
from spatialdata import SpatialData
from spatialdata._types import ArrayLike
from spatialdata.datasets import blobs
from spatialdata.models import Image2DModel, TableModel

from napari_spatialdata.utils._test_utils import save_image, take_screenshot

HERE: Path = Path(__file__).parent

SEED = 42

EXPECTED = HERE / "plots/groundtruth"
ACTUAL = HERE / "plots/generated"
TOL = 70
DPI = 40

RNG = np.random.default_rng(seed=0)
DATA_LEN = 100


def pytest_configure(config):
    config.DATA_LEN = 100


@pytest.fixture
def adata_labels() -> AnnData:
    n_var = 50

    blobs, _ = _get_blobs_galaxy()
    seg = np.unique(blobs)[1:]
    n_obs_labels = len(seg)
    rng = np.random.default_rng(SEED)

    obs_labels = pd.DataFrame(
        {
            "a": rng.normal(size=(n_obs_labels,)),
            "categorical": pd.Categorical(rng.integers(0, 2, size=(n_obs_labels,))),
            "cell_id": seg,
            "region": ["labels" for _ in range(n_obs_labels)],
        },
        index=np.arange(n_obs_labels),
    )
    uns_labels = {
        "spatial": {
            "labels": {
                "scalefactors": {
                    "spot_diameter_fullres": 10,
                    "tissue_hires_scalef": 1,
                    "tissue_segmentation_scalef": 1,
                }
            }
        }
    }
    obsm_labels = {"spatial": rng.integers(0, blobs.shape[0], size=(n_obs_labels, 2))}
    return TableModel.parse(
        generate_adata(n_var, obs_labels, obsm_labels, uns_labels),
        region="labels",
        region_key="region",
        instance_key="cell_id",
    )


@pytest.fixture
def annotation_values(adata_labels):
    """Generate random annotation values."""
    rng = np.random.default_rng()
    return rng.integers(0, 10, size=len(adata_labels.obs))


@pytest.fixture
def blobs_extra_cs() -> SpatialData:
    return blobs(extra_coord_system="space")


@pytest.fixture
def adata_shapes() -> AnnData:
    n_obs_shapes = 100
    n_var = 50
    blobs, _ = _get_blobs_galaxy()

    rng = np.random.default_rng(SEED)
    obs_shapes = pd.DataFrame(
        {
            "a": rng.normal(size=(n_obs_shapes,)),
            "categorical": pd.Categorical(rng.integers(0, 10, size=(n_obs_shapes,))),
        },
        index=np.arange(n_obs_shapes),
    )
    uns_shapes = {
        "spatial": {
            "shapes": {
                "scalefactors": {
                    "spot_diameter_fullres": 10,
                    "tissue_hires_scalef": 1,
                    "tissue_segmentation_scalef": 1,
                }
            }
        }
    }
    obsm_shapes = {"spatial": rng.integers(0, blobs.shape[0], size=(n_obs_shapes, 2))}
    return AnnData(
        rng.normal(size=(n_obs_shapes, n_var)),
        dtype=np.float64,
        obs=obs_shapes,
        obsm=obsm_shapes,
        uns=uns_shapes,
    )


@pytest.fixture()
def sdata_blobs() -> SpatialData:
    return blobs()


@pytest.fixture()
def sdata_channel_images() -> SpatialData:
    sdata = blobs()
    sdata["blobs_image_str_ch"] = Image2DModel.parse(
        sdata["blobs_image"], c_coords=["channel1", "channel2", "channel3"]
    )
    sdata["blobs_multiscale_image_str_ch"] = Image2DModel.parse(
        sdata["blobs_image"], c_coords=["channel1", "channel2", "channel3"], scale_factors=[2, 2]
    )
    return sdata


@pytest.fixture
def image():
    _, image = _get_blobs_galaxy()
    return image


@pytest.fixture
def labels():
    blobs, _ = _get_blobs_galaxy()
    return blobs


@pytest.fixture
def prepare_continuous_test_data():
    x_vec = RNG.random(DATA_LEN)
    y_vec = RNG.random(DATA_LEN)
    color_vec = RNG.random(DATA_LEN)

    x_data = {"vec": x_vec}
    y_data = {"vec": y_vec}
    color_data = {"vec": color_vec}

    x_label = generate_random_string(10)
    y_label = generate_random_string(10)
    color_label = generate_random_string(10)
    return x_data, y_data, color_data, x_label, y_label, color_label


@pytest.fixture
def prepare_discrete_test_data():
    x_vec = RNG.random(DATA_LEN)
    y_vec = RNG.random(DATA_LEN)
    color_vec = np.zeros(DATA_LEN).astype(int)

    x_data = {"vec": x_vec}
    y_data = {"vec": y_vec}
    color_data = {"vec": color_vec, "labels": ["a"]}

    x_label = generate_random_string(10)
    y_label = generate_random_string(10)
    color_label = generate_random_string(10)
    return x_data, y_data, color_data, x_label, y_label, color_label


def generate_random_string(length):
    letters = string.ascii_letters  # Includes both lowercase and uppercase letters
    return "".join(random.choice(letters) for i in range(length))


def _get_blobs_galaxy() -> tuple[ArrayLike, ArrayLike]:
    blobs = data.binary_blobs(rng=SEED)
    blobs = ndi.label(blobs)[0]
    return blobs, data.hubble_deep_field()[: blobs.shape[0], : blobs.shape[0]]


def generate_adata(n_var: int, obs: pd.DataFrame, obsm: dict[Any, Any], uns: dict[Any, Any]) -> AnnData:
    rng = np.random.default_rng(SEED)
    return AnnData(
        rng.normal(size=(obs.shape[0], n_var)),
        obs=obs,
        obsm=obsm,
        uns=uns,
        dtype=np.float64,
    )


class PlotTesterMeta(ABCMeta):
    def __new__(cls, clsname, superclasses, attributedict):
        for key, value in attributedict.items():
            if callable(value):
                attributedict[key] = _decorate(value, clsname, name=key)
        return super().__new__(cls, clsname, superclasses, attributedict)


# ideally, we would you metaclass=PlotTesterMeta and all plotting tests just subclass this
# but for some reason, pytest erases the metaclass info
class PlotTester(ABC):
    @classmethod
    def compare(cls, basename: str, tolerance: float | None = None):
        ACTUAL.mkdir(parents=True, exist_ok=True)
        out_path = ACTUAL / f"{basename}.png"

        viewer = napari.current_viewer()
        save_image(take_screenshot(viewer, canvas_only=True), out_path)

        if tolerance is None:
            # see https://github.com/theislab/squidpy/pull/302
            tolerance = 2 * TOL if "Napari" in str(basename) else TOL

        res = compare_images(str(EXPECTED / f"{basename}.png"), str(out_path), tolerance)

        assert res is None, res


def _decorate(fn: Callable, clsname: str, name: str | None = None) -> Callable:
    @wraps(fn)
    def save_and_compare(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        self.compare(fig_name)

    if not callable(fn):
        raise TypeError(f"Expected a `callable` for class `{clsname}`, found `{type(fn).__name__}`.")

    name = fn.__name__ if name is None else name

    if not name.startswith("test_plot_") or not clsname.startswith("Test"):
        return fn

    fig_name = f"{clsname[4:]}_{name[10:]}"

    return save_and_compare


@pytest.fixture
def caplog(caplog):
    handler_id = logger.add(caplog.handler, format="{message}")
    yield caplog
    logger.remove(handler_id)


@pytest.fixture(autouse=True)
def always_sync(monkeypatch, request):
    if request.node.get_closest_marker("use_thread_loader") is None:
        monkeypatch.setattr("napari_spatialdata._sdata_widgets.PROBLEMATIC_NUMPY_MACOS", True)
