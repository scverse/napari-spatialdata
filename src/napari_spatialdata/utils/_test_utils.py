from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qtpy.QtCore import Qt

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot
    from qtpy.QtCore import QPoint
    from qtpy.QtWidgets import QListWidget

from loguru import logger
from PIL import Image

from napari_spatialdata._interactive import Interactive
from napari_spatialdata.utils._utils import NDArrayA


def get_center_pos_listitem(widget: QListWidget, text: str) -> QPoint:
    """Get the center coordinates of a list item of a QListWidget based on text value.

    Parameters
    ----------
    widget: QListWidget
        The widget containing the list item of which the center position will be retrieved.
    text: str
        The text value of the list item for which the center position needs to be retrieved.

    Returns
    -------
    Qpoint
        The y and x center coordinates of a specific list item.
    """
    list_item = widget.findItems(text, Qt.MatchExactly)[0]
    model_index = widget.indexFromItem(list_item)
    return widget.visualRect(model_index).center()


def click_list_widget_item(qtbot: QtBot, widget: QListWidget, position: QPoint, wait_signal: str) -> None:
    """Simulate click on position in list widget.

    Parameters
    ----------
    qtbot
        Helper to simulate user input
    widget: QListWidget
        Widget containing list items
    position: QPoint
        The position y and x in the widget where the click event should take place.
    wait_signal: str
        Attribute of the widget which is a signal that should be waited for until send / received.
    """
    wait_signal = getattr(widget, wait_signal)
    with qtbot.wait_signal(wait_signal):
        qtbot.mouseClick(
            widget.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            pos=position,
        )


def take_screenshot(interactive: Interactive) -> NDArrayA | Any:
    """Take screenshot of interactive viewer.

    Parameters
    ----------
    interactive: Interactive
        Interactive object containing the viewer.
    """
    logger.info("Taking screenshot of viewer")
    # to distinguish between the black of the image background and the black of the napari background (now white)
    interactive._viewer.theme = "light"
    interactive_screenshot = interactive._viewer.screenshot(canvas_only=False)
    interactive._viewer.theme = "dark"
    interactive._viewer.close()

    return interactive_screenshot


def save_image(image_np: NDArrayA, file_name: str) -> None:
    """Save image to file. This was used to generate tests/plots/plot_image.png.

    Parameters
    ----------
    image_np: NDArrayA
        Image as numpy array.
    """
    im = Image.fromarray(image_np)
    im.save(file_name)  # Saving in lossless format
