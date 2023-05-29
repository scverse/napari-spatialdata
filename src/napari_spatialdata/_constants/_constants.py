"""Constants that user deals with."""
from enum import Enum, unique

from napari.layers import Image, Labels, Points, Shapes

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


@unique
class SpatialDataLayers(Enum):
    images = Image
    points = Points
    labels = Shapes
    shapes = Labels
