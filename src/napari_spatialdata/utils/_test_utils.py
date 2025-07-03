from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from qtpy.QtCore import Qt

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot
    from qtpy.QtCore import QPoint
    from qtpy.QtWidgets import QListWidget

import napari
from loguru import logger
from PIL import Image
from spatialdata._types import ArrayLike


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


def click_list_widget_item(
    qtbot: QtBot, widget: QListWidget, position: QPoint, wait_signal: str, click: Literal["single", "double"] = "single"
) -> None:
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
        if click == "single":
            qtbot.mouseClick(
                widget.viewport(),
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                pos=position,
            )
        elif click == "double":
            qtbot.mouseDClick(
                widget.viewport(),
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
                pos=position,
            )
        else:
            raise ValueError(f"{click} is not a valid click")


def export_figure(viewer: napari.Viewer) -> ArrayLike | Any:
    """Export figure from the data displayed in the Napari viewer.

    Parameters
    ----------
    viewer
        Instance of napari Viewer.

    Returns
    -------
    The exported figure as an array.
    """
    logger.info("Export figure of data in viewer canvas")
    old_theme = viewer.theme
    viewer.theme = "light"
    interactive_screenshot = viewer.export_figure()
    viewer.theme = old_theme
    viewer.close()

    return interactive_screenshot


def save_image(image_np: ArrayLike, file_path: str) -> None:
    """Save image to file.

    Parameters
    ----------
    image_np
        Image as numpy array.
    file_path
        File path of the image.
    """
    im = Image.fromarray(image_np)
    im.save(file_path)
