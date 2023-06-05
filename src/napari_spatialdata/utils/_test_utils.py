from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt

if TYPE_CHECKING:
    from PyQt6.QtCore import QPoint
    from PyQt6.QtWidgets import QListWidget


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
