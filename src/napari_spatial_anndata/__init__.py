
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


from ._reader import napari_get_reader
from ._writer import write_single_image, write_multiple 
from ._sample_data import make_sample_data 
from ._widget import ExampleQWidget, example_magic_widget
