from abc import ABC, ABCMeta
from typing import Callable, Optional
from pathlib import Path
from functools import wraps

from matplotlib.testing.compare import compare_images
import pytest
import matplotlib.pyplot as plt

HERE: Path = Path(__file__).parent

EXPECTED = HERE / "_images"
ACTUAL = HERE / "figures"
TOL = 50
DPI = 40

C_KEY_PALETTE = "leiden"


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


def pytest_addoption(parser):
    parser.addoption("--test-napari", action="store_true", help="Test interactive image view")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--test-napari"):
        return
    skip_slow = pytest.mark.skip(reason="Need --test-napari option to test interactive image view")
    for item in items:
        if "qt" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def _test_napari(pytestconfig):
    _ = pytestconfig.getoption("--test-napari", skip=True)


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
