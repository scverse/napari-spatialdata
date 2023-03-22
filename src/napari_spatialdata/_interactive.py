from typing import Union, Iterable

from spatialdata import SpatialData
from napari.viewer import Viewer
from qtpy.QtWidgets import QLabel, QWidget, QListWidget, QVBoxLayout, QListWidgetItem
import napari


class ElementWidget(QListWidget):
    def __init__(self, sdata: SpatialData):
        super().__init__()
        self._sdata = sdata

    def _onClickChange(self, selected_coordinate_system: Union[QListWidgetItem, int, Iterable[str]]) -> None:
        self.clear()

        element_names = []
        for _, element_name, _ in self._sdata.filter_by_coordinate_system(selected_coordinate_system)._gen_elements():
            element_names.append(element_name)

        self.addItems(element_names)


class CoordinateSystemWidget(QListWidget):
    def __init__(self, sdata: SpatialData):
        super().__init__()

        self._sdata = sdata

        self.addItems(self._sdata.coordinate_systems)


class SdataWidget(QWidget):
    def __init__(self, viewer: Viewer, sdata: SpatialData):
        super().__init__()
        self._sdata = sdata
        self._viewer = viewer

        self.setLayout(QVBoxLayout())

        self.coordinate_system_widget = CoordinateSystemWidget(self._sdata)
        self.elements_widget = ElementWidget(self._sdata)

        self.layout().addWidget(QLabel("Coordinate System:"))
        self.layout().addWidget(self.coordinate_system_widget)
        self.layout().addWidget(QLabel("Elements:"))
        self.layout().addWidget(self.elements_widget)

        self.elements_widget.itemDoubleClicked.connect(lambda item: self._onClick(item.text()))
        self.coordinate_system_widget.itemClicked.connect(lambda item: self.elements_widget._onClickChange(item.text()))

    def _onClick(self, text: str) -> None:
        if "labels" in text:
            self._add_label(text)
        elif "image" in text:
            self._add_image(text)
        elif "points" in text:
            raise NotImplementedError("Points is currently not supported due to performance issues!")

    def _add_label(self, key: str) -> None:
        self._viewer.add_labels(
            self._sdata.labels[key],
            name=key,
            metadata={
                "adata": self._sdata.table[
                    self._sdata.table.obs[self._sdata.table.uns["spatialdata_attrs"]["region_key"]] == key
                ],
                "labels_key": self._sdata.table.uns["spatialdata_attrs"]["instance_key"],
            },
        )

    def _add_image(self, key: str) -> None:
        self._viewer.add_image(
            self._sdata.images[key],
            name=key,
            metadata={
                "adata": self._sdata.table[
                    self._sdata.table.obs[self._sdata.table.uns["spatialdata_attrs"]["region_key"]] == key
                ],
                "labels_key": self._sdata.table.uns["spatialdata_attrs"]["instance_key"],
            },
        )

    def _add_point(self, key: str) -> None:
        self._viewer.add_points(
            self._sdata.points[key],
            name=key,
        )

        # TODO magicgui update in plugins --possible?


class Interactive:
    def __init__(self, sdata: SpatialData):
        self._viewer = napari.Viewer()
        self._sdata = sdata
        self._sdata_widget = SdataWidget(self._viewer, sdata)
        self._list_widget = self._viewer.window.add_dock_widget(
            self._sdata_widget, name="SpatialData", area="left", menu=self._viewer.window.window_menu
        )

    def run(self) -> None:
        napari.run()
