from abc import ABC, ABCMeta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from loguru import logger
from matplotlib.testing.compare import compare_images
from napari_spatialdata._utils import NDArrayA
from scipy import ndimage as ndi
from skimage import data

HERE: Path = Path(__file__).parent

SEED = 42

EXPECTED = HERE / "_images"
ACTUAL = HERE / "figures"
TOL = 50
DPI = 40


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
            "cell_id": pd.Categorical(seg),
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
    return generate_adata(n_var, obs_labels, obsm_labels, uns_labels)


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


@pytest.fixture
def image():
    _, image = _get_blobs_galaxy()
    return image


@pytest.fixture
def labels():
    blobs, _ = _get_blobs_galaxy()
    return blobs


def _get_blobs_galaxy() -> Tuple[NDArrayA, NDArrayA]:
    blobs = data.binary_blobs(seed=SEED)
    blobs = ndi.label(blobs)[0]
    return blobs, data.hubble_deep_field()[: blobs.shape[0], : blobs.shape[0]]


def generate_adata(n_var: int, obs: pd.DataFrame, obsm: Dict[Any, Any], uns: Dict[Any, Any]) -> AnnData:
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
    def compare(cls, basename: str, tolerance: Optional[float] = None):
        ACTUAL.mkdir(parents=True, exist_ok=True)
        out_path = ACTUAL / f"{basename}.png"

        plt.savefig(out_path, dpi=DPI)
        plt.close()

        if tolerance is None:
            # see https://github.com/theislab/squidpy/pull/302
            tolerance = 2 * TOL if "Napari" in str(basename) else TOL

        res = compare_images(str(EXPECTED / f"{basename}.png"), str(out_path), tolerance)

        assert res is None, res


def _decorate(fn: Callable, clsname: str, name: Optional[str] = None) -> Callable:
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
