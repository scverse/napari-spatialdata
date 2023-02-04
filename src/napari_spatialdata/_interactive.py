from spatialdata import SpatialData
from napari.viewer import Viewer
from qtpy.QtWidgets import QListWidget
import napari


class SdataWidget(QListWidget):
    def __init__(self, viewer: Viewer, sdata: SpatialData):
        super().__init__()
        self._sdata = sdata
        self._viewer = viewer
        self.addItems()
        self.itemDoubleClicked.connect(lambda item: self._add_image(item.text()))

    def addItems(self) -> None:
        texts = self._sdata.labels.keys()
        super().addItems(tuple(texts))

    def _add_image(self, key: str) -> None:
        self._viewer.add_labels(
            self._sdata.labels[key],
            name=key,
            metadata={
                "adata": self._sdata.table[
                    self._sdata.table.obs[sdata.table.uns["spatialdata_attrs"]["region_key"]] == key
                ],
                "library_id": key,  # ???? TODO fix for view
                "labels_key": self._sdata.table.uns["spatialdata_attrs"]["instance_key"],
            },
        )
        # TODO magicgui update in plugins --possible?


class Interactive:
    def __init__(self, sdata: SpatialData):
        self._viewer = napari.Viewer()
        self._sdata = sdata
        self._sdata_widget = SdataWidget(self._viewer, sdata)
        self._list_widget = self._viewer.window.add_dock_widget(
            self._sdata_widget, name="Select spatialdata segment", area="left"
        )  # TODO fix list so it shows everything
        # TODO add to window
        napari.run()


if __name__ == "__main__":  # TODO: create example instead of this
    sdata = SpatialData.read("../cosmx.zarr")
    sdata.table.uns["spatialdata_attrs"]["region"] = 0
    Interactive(sdata)
