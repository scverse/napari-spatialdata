"""Constants that user deals with."""
from enum import unique

from napari_spatialdata._constants._utils import ModeEnum


@unique
class Symbol(str, ModeEnum):
    DISC = "disc"
    SQUARE = "square"


@unique
class InferDimensions(ModeEnum):
    DEFAULT = "default"
    CHANNELS_LAST = "channels_last"
    Z_LAST = "z_last"
