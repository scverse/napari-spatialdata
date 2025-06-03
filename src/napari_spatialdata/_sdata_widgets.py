"""Widgets for displaying and interacting with SpatialData objects in napari.

This module provides a set of Qt widgets for visualizing and interacting with
SpatialData objects within the napari viewer. It includes a ListWidget for selecting
coordinate systems, browsing elements within SpatialData objects, and handling
channel selection for multidimensional image data.
"""

from __future__ import annotations

import platform
from collections.abc import Iterable
from importlib.metadata import version
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import shapely
from napari.layers import Points, Shapes
from napari.utils.events import EventedList
from napari.utils.notifications import show_info
from packaging.version import parse as parse_version
from qtpy.QtCore import QThread, Signal
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QLabel, QListWidget, QListWidgetItem, QProgressBar, QVBoxLayout, QWidget
from spatialdata import SpatialData
from spatialdata.models._utils import DEFAULT_COORDINATE_SYSTEM
from xarray import DataArray, DataTree

from napari_spatialdata._viewer import SpatialDataViewer
from napari_spatialdata.constants.config import N_CIRCLES_WARNING_THRESHOLD, N_SHAPES_WARNING_THRESHOLD
from napari_spatialdata.utils._utils import (
    _get_sdata_key,
    get_duplicate_element_names,
    get_elements_meta_mapping,
)

if TYPE_CHECKING:
    from napari import Viewer
    from napari.utils.events.event import Event

icon_path = Path(__file__).parent / "resources/exclamation.png"

# if run with numpy<2 on macOS arm64 architecture compiled from pypi wheels,
# then it will crash with bus error if numpy is used in different thread
# Issue reported https://github.com/numpy/numpy/issues/21799
if (
    parse_version(version("napari")) < parse_version("0.5.3")
    and parse_version(version("numpy")) < parse_version("2")
    and platform.system() == "Darwin"
    and platform.machine() == "arm64"
):  # pragma: no cover
    try:
        PROBLEMATIC_NUMPY_MACOS = "cibw-run" in np.show_config("dicts")["Python Information"]["path"]  # type: ignore[call-arg,func-returns-value,unused-ignore]
    except (KeyError, TypeError):
        PROBLEMATIC_NUMPY_MACOS = False
else:
    PROBLEMATIC_NUMPY_MACOS = False


class ListWidget(QListWidget):
    """Widget for displaying and selecting coordinate systems or elements from SpatialData objects or channels.

    This widget can show a list of coordinate systems or available elements (images, labels, points, shapes)
    from the SpatialData objects, with warnings for elements that might be slow to render. A third option is to
    let it show channels from image elements.

    The widget's behavior is determined by the `widget_type` parameter passed during initialization:
    - "coordinate_system": Displays available coordinate systems from SpatialData objects
    - "element": Displays available elements (images, labels, points, shapes) from SpatialData objects
    - "channel": Displays available channels from selected image elements

    Attributes
    ----------
    _widget_type : str
        Type of widget ("coordinate_system", "element", or "channel") determining its behavior.
    _icon : QIcon
        Icon used for warning indicators for elements that might be slow to render.
    _sdata : EventedList
        List of SpatialData objects.
    _duplicate_element_names : dict
        Dictionary of duplicate element names across SpatialData objects.
    _element_widget_text : str or None
        Text of the currently selected element in the ElementWidget.
    _element_dict : dict or None
        Dictionary with metadata of the currently selected element.
    _system : str or None
        Currently selected coordinate system.
    """

    def __init__(self, sdata: EventedList, coordinate_system: bool = False):
        """Initialize the Widget.

        Parameters
        ----------
        sdata : EventedList
            List of SpatialData objects to display elements from.
        widget_type: Literal["coordinate_system", "element", "channel"]
            The type of the widget. This determines what kind of items it will show.
        """
        super().__init__()
        self._icon = QIcon(str(icon_path))
        self._sdata = sdata
        self._duplicate_element_names, _ = get_duplicate_element_names(self._sdata)
        self._element_widget_text: str | None = None
        self._elements: dict[str, dict[str, str | int]] | None = None
        self._system: None | str = None

        if coordinate_system:
            # Sort alphabetically, but keep default "global" at the top.
            coordinate_systems = sorted({cs for sdata in self._sdata for cs in sdata.coordinate_systems})
            if DEFAULT_COORDINATE_SYSTEM in coordinate_systems:
                coordinate_systems.remove(DEFAULT_COORDINATE_SYSTEM)
                coordinate_systems.insert(0, DEFAULT_COORDINATE_SYSTEM)
            self.addItems(coordinate_systems)

    def _onCsItemChange(self, selected_coordinate_system: QListWidgetItem | int | Iterable[str]) -> None:
        """Update the element list of an element widget when the coordinate system selection changes.

        Parameters
        ----------
        selected_coordinate_system : QListWidgetItem or int or Iterable[str]
            The newly selected coordinate system.
            Can be a QListWidgetItem, an index, or an iterable of strings.
        """
        self.clear()
        elements, _ = get_elements_meta_mapping(self._sdata, selected_coordinate_system, self._duplicate_element_names)
        self._set_element_widget_items(elements)
        self._elements = elements

    def _set_element_widget_items(self, elements: dict[str, dict[str, str | int]]) -> None:
        """Populate an element widget with element items.

        Adds each element as an item in the list widget, with warning icons for elements
        that might be slow to render (e.g., many circles or shapes).

        Parameters
        ----------
        elements : dict[str, dict[str, str | int]]
            Dictionary mapping element names to their metadata.
        """
        for key, dict_val in sorted(elements.items(), key=itemgetter(0)):
            sdata = self._sdata[dict_val["sdata_index"]]
            element_type = dict_val["element_type"]
            element_name = dict_val["original_name"]
            item = QListWidgetItem(key)
            if element_type == "shapes":
                if (
                    type((element := sdata.shapes[element_name]).iloc[0].geometry) is shapely.Point
                    and len(element) > N_CIRCLES_WARNING_THRESHOLD
                ):
                    item.setIcon(self._icon)
                    item.setToolTip(
                        "Visualizing this many circles is currently slow in napari. Consider whether you want to "
                        "visualize."
                    )
                    self.addItem(item)
                elif (
                    type((element := sdata.shapes[element_name]).iloc[0].geometry)
                    in [shapely.Polygon, shapely.MultiPolygon]
                    and len(element) > N_SHAPES_WARNING_THRESHOLD
                ):
                    item.setIcon(self._icon)
                    item.setToolTip(
                        "Visualizing this many shapes is currently slow in napari. Consider whether you want to "
                        "visualize."
                    )
            self.addItem(item)

    def _select_coord_sys(self, selected_coordinate_system: QListWidgetItem | int | Iterable[str]) -> None:
        """Store the currently selected coordinate system.

        Parameters
        ----------
        selected_coordinate_system : QListWidgetItem or int or Iterable[str]
            The selected coordinate system.
            Can be a QListWidgetItem, an index, or an iterable of strings.
        """
        self._system = str(selected_coordinate_system)

    def _on_element_item_changed(
        self, sdata: SpatialData, element_widget_text: str, element_dict: dict[str, str | int]
    ) -> None:
        """Update the channel items in the channel widget when the selected element changes.

        Clears the current channel list and populates it with channels from the
        selected element if it's an image.

        Parameters
        ----------
        sdata : SpatialData
            The SpatialData object containing the selected element.
        element_widget_text : str
            Text of the selected element in the ElementWidget.
        element_dict : dict
            Dictionary with metadata of the selected element.
        """
        self.clear()
        self._element_widget_text = element_widget_text
        if element_dict["element_type"] == "images":
            element: DataArray | DataTree = sdata[element_dict["original_name"]]
            self._element_widget_text = element_widget_text
            self._set_channel_widget_items(element)

    def _set_channel_widget_items(self, element: DataArray | DataTree) -> None:
        """Populate a channel widget with channel items from the selected image element.

        Adds each channel as an item in the list widget, except for RGB(A) channels
        which are handled differently.

        Parameters
        ----------
        element : object
            The image element to extract channels from.
        """
        if isinstance(element, DataArray):
            channels = list(element.c.to_numpy())
        else:
            channels = list(element["scale0"].c.to_numpy())

        if channels not in [["r", "g", "b"], ["r", "g", "b", "a"]]:
            for ch in channels:
                item = QListWidgetItem(str(ch))
                self.addItem(item)


class DataLoadThread(QThread):
    """Thread for asynchronously loading SpatialData elements.

    This thread handles loading different types of data (images, labels, points, shapes)
    from SpatialData objects without blocking the UI.

    Parameters
    ----------
    parent : SdataWidget
        Parent SdataWidget that owns this thread.

    Attributes
    ----------
    returned : Signal
        Signal emitted when data loading is complete, carrying the created layer.
    sdata_widget : SdataWidget
        Parent SdataWidget that owns this thread.
    _data_type : str
        Type of data to load (images, labels, points, shapes).
    _text : str
        Name of the element to load.
    _sdata : SpatialData
        SpatialData object containing the element.
    _selected_cs : str
        Selected coordinate system.
    _multi : bool
        Boolean indicating if multiple SpatialData objects are present.
    _channel_name : str, optional
        Optional channel name for image data.
    """

    returned = Signal(object)

    def __init__(self, parent: SdataWidget):
        """Initialize the DataLoadThread.

        Parameters
        ----------
        parent : SdataWidget
            Parent SdataWidget that owns this thread.
        """
        super().__init__(parent=parent)
        self.sdata_widget = parent
        self._data_type = ""
        self._text = ""
        self._sdata = None
        self._selected_cs: str = ""
        self._multi: bool = False

    def load_data(
        self,
        data_type: str,
        text: str,
        sdata: SpatialData,
        selected_cs: str,
        multi: bool,
        channel_name: str | None = None,
    ) -> None:
        """Set up data loading parameters and start the thread.

        Parameters
        ----------
        data_type : str
            Type of data to load (images, labels, points, shapes).
        text : str
            Name of the element to load.
        sdata : SpatialData
            SpatialData object containing the element.
        selected_cs : str
            Selected coordinate system.
        multi : bool
            Boolean indicating if multiple SpatialData objects are present.
        channel_name : str, optional
            Optional channel name for image data.

        Raises
        ------
        RuntimeError
            If the thread is already running.
        """
        if self.isRunning():
            raise RuntimeError("Thread is already running.")
        self._data_type = data_type
        self._text = text
        self._channel_name = channel_name
        self._sdata = sdata
        self._selected_cs = selected_cs
        self._multi = multi

        if PROBLEMATIC_NUMPY_MACOS:
            self.run()
        else:
            self.start()

    def run(self) -> None:
        """Execute the data loading operation.

        Loads the specified data element based on its type and emits the
        returned layer through the 'returned' signal.
        """
        if not self._data_type:
            return
        if self._data_type == "labels":
            layer = self.sdata_widget.viewer_model.get_sdata_labels(
                self._sdata, self._text, self._selected_cs, self._multi
            )
        elif self._data_type == "images":
            layer = self.sdata_widget.viewer_model.get_sdata_image(
                self._sdata, self._text, self._selected_cs, self._multi, self._channel_name
            )
        elif self._data_type == "points":
            layer = self.sdata_widget.viewer_model.get_sdata_points(
                self._sdata, self._text, self._selected_cs, self._multi
            )
        elif self._data_type == "shapes":
            layer = self.sdata_widget._get_shapes(self._sdata, self._text, self._selected_cs, self._multi)
        else:
            raise ValueError(f"Data type {self._data_type} not recognized.")

        self.returned.emit(layer)


class SdataWidget(QWidget):
    """Main widget for interacting with SpatialData objects in napari.

    This widget combines coordinate system selection, element browsing, and channel
    selection into a unified interface for visualizing SpatialData objects in napari.
    It manages the loading and display of different data types and handles coordinate
    system transformations.

    Attributes
    ----------
    _sdata
        List of SpatialData objects.
    viewer_model
        SpatialDataViewer instance for interacting with napari.
    worker_thread
        Thread for asynchronous data loading.
    coordinate_system_widget
        Widget for selecting coordinate systems.
    elements_widget
        Widget for browsing and selecting elements.
    channel_widget
        Widget for selecting channels in image data.
    slider
        Progress bar shown during data loading.
    """

    def __init__(self, viewer: Viewer, sdata: EventedList):
        """Initialize the SdataWidget.

        Parameters
        ----------
        viewer : Viewer
            napari Viewer instance.
        sdata : EventedList
            List of SpatialData objects to visualize.
        """
        super().__init__()
        self._sdata = sdata
        self.viewer_model = SpatialDataViewer(viewer, self._sdata)
        self.worker_thread = DataLoadThread(parent=self)
        self.worker_thread.returned.connect(self.viewer_model.viewer.add_layer)
        self.worker_thread.finished.connect(self._hide_slider)

        self.setLayout(QVBoxLayout())

        self.coordinate_system_widget = ListWidget(self._sdata, coordinate_system=True)
        self.elements_widget = ListWidget(self._sdata)
        self.channel_widget = ListWidget(self._sdata)
        self.slider = QProgressBar(self)
        self.slider.setRange(0, 0)
        self.slider.setVisible(False)

        self.layout().addWidget(self.slider)
        self.layout().addWidget(QLabel("Coordinate System:"))
        self.layout().addWidget(self.coordinate_system_widget)
        self.layout().addWidget(QLabel("Elements:"))
        self.layout().addWidget(self.elements_widget)
        self.layout().addWidget(QLabel("Channels:"))
        self.layout().addWidget(self.channel_widget)
        self.elements_widget.currentItemChanged.connect(self._on_element_item_changed)
        self.elements_widget.itemDoubleClicked.connect(self._on_doubleclick_element_item)
        self.channel_widget.itemDoubleClicked.connect(self._on_doubleclick_channel_item)
        self.coordinate_system_widget.currentItemChanged.connect(
            lambda item: self.elements_widget._onCsItemChange(item.text())
        )
        self.coordinate_system_widget.currentItemChanged.connect(
            lambda item: self.coordinate_system_widget._select_coord_sys(item.text())
        )
        self.viewer_model.layer_saved.connect(self.elements_widget._onCsItemChange)
        self.coordinate_system_widget.currentItemChanged.connect(self._update_layers_visibility)
        self.coordinate_system_widget.currentItemChanged.connect(
            lambda item: self.viewer_model._affine_transform_layers(item.text())
        )
        self.viewer_model.viewer.layers.events.inserted.connect(self._on_insert_layer)

    def _on_insert_layer(self, event: Event) -> None:
        """Connect visibility events for newly inserted layers.

        Parameters
        ----------
        event : Event
            Event containing the newly inserted layer.
        """
        layer = event.value
        layer.events.visible.connect(self._update_visible_in_coordinate_system)

    def _on_element_item_changed(self, item: QListWidgetItem) -> None:
        """Handle selection changes in the elements widget.

        Updates the channel widget with channels from the selected element.

        Parameters
        ----------
        item : QListWidgetItem
            The newly selected element item.
        """
        if self.elements_widget._elements:
            sdata, _ = _get_sdata_key(self._sdata, self.elements_widget._elements, item.text())
            self.channel_widget._on_element_item_changed(
                sdata, item.text(), self.elements_widget._elements[item.text()]
            )

    def _on_doubleclick_channel_item(self, item: QListWidgetItem) -> None:
        """Handle double-click events on channel items in the channel widget.

        Loads and displays the selected channel of the current element.

        Parameters
        ----------
        item : QListWidgetItem
            The double-clicked channel item.
        """
        if self.channel_widget._element_widget_text:
            self._onClick(self.channel_widget._element_widget_text, item.text())

    def _on_doubleclick_element_item(self, item: QListWidgetItem) -> None:
        """Handle double-click events on element items in the element widget.

        Loads and displays the selected element.

        Parameters
        ----------
        item : QListWidgetItem
            The double-clicked element item.
        """
        self._onClick(item.text())

    def _hide_slider(self) -> None:
        """Hide the progress slider when data loading is complete."""
        self.slider.setVisible(False)

    def _onClick(self, element_name: str, channel_name: str | None = None) -> None:
        """Handle click events to load and display data elements.

        Parameters
        ----------
        element_name : str
            Name of the element to load.
        channel_name : str, optional
            Name of the channel to load for image elements.
        """
        selected_cs = self.coordinate_system_widget._system
        if self.worker_thread.isRunning():
            show_info("Please wait for the current operation to finish.")
            return

        if selected_cs and self.elements_widget._elements:
            sdata, multi = _get_sdata_key(self._sdata, self.elements_widget._elements, element_name)
            if (type_ := self.elements_widget._elements[element_name]["element_type"]) not in {
                "labels",
                "images",
                "shapes",
                "points",
            }:
                return

            type_ = cast(str, type_)

            self.worker_thread.load_data(type_, element_name, sdata, selected_cs, multi, channel_name)
            if not PROBLEMATIC_NUMPY_MACOS:
                self.slider.setVisible(True)

    def _update_visible_in_coordinate_system(self, event: Event) -> None:
        """Toggle active status in the coordinate system metadata when changing layer visibility.

        Parameters
        ----------
        event : Event
            Event triggered by changing layer visibility.
        """
        metadata = event.source.metadata
        layer_active = metadata.get("_active_in_cs")
        selected_coordinate_system = self.coordinate_system_widget._system

        elements = self.elements_widget._elements
        element_name = metadata.get("name")
        if elements and element_name and element_name in elements:
            if selected_coordinate_system not in layer_active:
                layer_active.add(selected_coordinate_system)
            else:
                layer_active.remove(selected_coordinate_system)

    def _update_layers_visibility(self) -> None:
        """Toggle layer visibility based on presence in the currently selected coordinate system.

        Updates the visibility of all layers based on whether they are active in the
        currently selected coordinate system. Also updates layer metadata to track
        coordinate system information.
        """
        elements = self.elements_widget._elements
        coordinate_system = self.coordinate_system_widget._system
        # No layer selected on first time coordinate system selection
        if self.viewer_model.viewer.layers:
            for layer in self.viewer_model.viewer.layers:
                element_name = layer.metadata.get("name")
                if element_name:
                    if elements and (
                        layer.name not in elements or element_name != elements[layer.name]["original_name"]
                    ):
                        layer.visible = False
                    elif layer.metadata["_active_in_cs"]:
                        layer.visible = True
                        # Prevent _update_visible_in_coordinate_system of invalid removal of coordinate system
                        layer.metadata["_active_in_cs"].add(coordinate_system)
                        layer.metadata["_current_cs"] = coordinate_system

    def _get_shapes(self, sdata: SpatialData, key: str, selected_cs: str, multi: bool) -> Shapes | Points:
        """Load and create appropriate layer for shape data.

        Determines the geometry type of the shapes element and calls the appropriate
        method to create either a Points layer (for Point geometries) or a Shapes
        layer (for Polygon or MultiPolygon geometries).

        Parameters
        ----------
        sdata : SpatialData
            SpatialData object containing the shapes element.
        key : str
            Name of the shapes element to load.
        selected_cs : str
            Selected coordinate system.
        multi : bool
            Whether multiple SpatialData objects are present.

        Returns
        -------
        Shapes or Points
            The created napari layer.

        Raises
        ------
        TypeError
            If the geometry type is not Point, Polygon, or MultiPolygon.
        """
        original_name = key[: key.rfind("_")] if multi else key

        if type(sdata.shapes[original_name].iloc[0].geometry) is shapely.geometry.point.Point:
            return self.viewer_model.get_sdata_circles(sdata, key, selected_cs, multi)
        if (type(sdata.shapes[original_name].iloc[0].geometry) is shapely.geometry.polygon.Polygon) or (
            type(sdata.shapes[original_name].iloc[0].geometry) is shapely.geometry.multipolygon.MultiPolygon
        ):
            return self.viewer_model.get_sdata_shapes(sdata, key, selected_cs, multi)

        raise TypeError("Incorrect data type passed for shapes (should be Shapely Point or Polygon or MultiPolygon).")
