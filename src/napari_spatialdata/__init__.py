__version__ = "0.1.0"

from importlib.metadata import version  # Python = 3.9

from packaging.version import parse

try:
    __full_version__ = parse(version(__name__))
    __full_version__ = f"{__version__}+{__full_version__.local}" if __full_version__.local else __version__
except ImportError:
    __full_version__ = __version__

del version, parse

from napari_spatialdata._interactive import Interactive as Interactive  # noqa: E402
from napari_spatialdata._view import (  # noqa: E402
    QtAdataScatterWidget,
    QtAdataViewWidget,
)
