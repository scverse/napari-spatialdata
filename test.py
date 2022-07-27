from loguru import logger
from scanpy import logging as logg
import numpy as np

from src.napari_spatialdata._utils import (
    _min_max_norm,
)


def log():
    logger.debug(f"Loguru: Called log debug function")
    logger.warning(f"Loguru: Called log warning function")
    logger.info(f"Loguru: Called log info function.")

    logg.warning("Scanpy: Called log warning function")


def main():

    _min_max_norm(np.array([0, 0, 2]))


if __name__ == "__main__":
    main()
