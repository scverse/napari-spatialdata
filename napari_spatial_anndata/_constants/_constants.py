"""Constants that user deals with."""
from enum import unique

from napari_spatial_anndata._constants._utils import ModeEnum


@unique
class Symbol(str, ModeEnum):
    DISC = "disc"
    SQUARE = "square"
