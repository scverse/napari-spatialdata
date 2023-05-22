from __future__ import annotations

from typing import Callable

from loguru import logger
from napari_plugin_engine import napari_hook_implementation
from spatialdata import SpatialData

from napari_spatialdata import Interactive

readable_extensions = (".zarr",)


@napari_hook_implementation
def get_reader(path: str) -> Callable[..., list[tuple[None]]] | None:
    """Napari hook specification that start the napari-spatialdata plugin with the given path."""
    # if we know we cannot read the file, we immediately return None.
    if not path.endswith(readable_extensions):
        return None
    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(path: str) -> list[tuple[None]]:
    """Reader function to load plugin with a given path.

    Take a path to a .zarr folder, load the napari-spatialdata plugin with it and
    returns an empty dummy object with no layer.
    """
    logger.info(f"Reading {path}")

    # use Interactive class to load plugin
    sdata = SpatialData.read(path)  # Change this path!
    _ = Interactive(sdata)

    # Readers are expected to return data as a list of tuples, where each tuple
    # is (data, [meta_dict, [layer_type]])
    # Make dummy object to return no layers
    return [(None,)]
