import numpy as np
from napari_spatialdata._interactive import Interactive
from napari_spatialdata.utils._test_utils import take_screenshot
from PIL import Image
from spatialdata.datasets import blobs


def test_interactive_add_image():
    sdata_blobs = blobs()

    i = Interactive(sdata=sdata_blobs, headless=True)
    i.add_element(coordinate_system_name="global", elements="blobs_image")
    screenshot = take_screenshot(i)

    # Load presaved image and compare with screenshoted image
    image = Image.open("tests/plots/plot_image.tiff")
    presaved_screenshot = np.array(image)

    assert np.array_equal(screenshot, presaved_screenshot)
