from __future__ import annotations

from typing import Any, Iterable

import napari
import numpy as np
import shapely
from anndata import AnnData
from loguru import logger
from multiscale_spatial_image import MultiscaleSpatialImage
from napari.viewer import Viewer
from qtpy.QtWidgets import QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget
from spatialdata import SpatialData

from napari_spatialdata._utils import NDArrayA, _get_transform, _swap_coordinates


class ElementWidget(QListWidget):
    def __init__(self, sdata: SpatialData):
        super().__init__()
        self._sdata = sdata

    def _onClickChange(self, selected_coordinate_system: QListWidgetItem | int | Iterable[str]) -> None:
        self.clear()

        elements = {}
        for element_type, element_name, _ in self._sdata.filter_by_coordinate_system(
            selected_coordinate_system
        )._gen_elements():
            elements[element_name] = element_type

        self.addItems(elements.keys())
        self._elements = elements


class CoordinateSystemWidget(QListWidget):
    def __init__(self, sdata: SpatialData):
        super().__init__()

        self._sdata = sdata

        self.addItems(self._sdata.coordinate_systems)

    def _select_coord_sys(self, selected_coordinate_system: QListWidgetItem | int | Iterable[str]) -> None:
        self._system = str(selected_coordinate_system)


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
        self.coordinate_system_widget.itemClicked.connect(
            lambda item: self.coordinate_system_widget._select_coord_sys(item.text())
        )

    def _onClick(self, text: str) -> None:
        if self.elements_widget._elements[text] == "labels":
            self._add_label(text)
        elif self.elements_widget._elements[text] == "images":
            self._add_image(text)
        elif self.elements_widget._elements[text] == "points":
            self._add_points(text)
        elif self.elements_widget._elements[text] == "shapes":
            self._add_shapes(text)

    def _add_circles(self, key: str) -> None:
        circles = []
        df = self._sdata.shapes[key]
        affine = _get_transform(self._sdata.shapes[key], self.coordinate_system_widget._system)

        for i in range(0, len(df)):
            circles.append([df.geometry[i].coords[0], [df.radius[i], df.radius[i]]])

        circles = _swap_coordinates(circles)

        self._viewer.add_shapes(
            circles,
            name=key,
            affine=affine,
            shape_type="ellipse",
            metadata={
                "adata": self._sdata.table[
                    self._sdata.table.obs[self._sdata.table.uns["spatialdata_attrs"]["region_key"]] == key
                ],
                "shapes_key": self._sdata.table.uns["spatialdata_attrs"]["region_key"],
                "shapes_type": "circles",
            },
        )

    def _add_polygons(self, key: str) -> None:
        polygons = []
        df = self._sdata.shapes[key]
        affine = _get_transform(self._sdata.shapes[key], self.coordinate_system_widget._system)

        # when mulitpolygons are present, we select the largest ones
        if "MultiPolygon" in np.unique(df.geometry.type):
            logger.info("Multipolygons are present in the data. Only the largest polygon per cell is retained.")
            df = df.explode(index_parts=False)
            df["area"] = df.area
            df = df.sort_values(by="area", ascending=False)  # sort by area
            df = df[~df.index.duplicated(keep="first")]  # only keep the largest area
            df = df.sort_index()  # reset the index to the first order
        if len(df) < 100:
            for i in range(0, len(df)):
                polygons.append(list(df.geometry.iloc[i].exterior.coords))
        else:
            for i in range(
                0, len(df)
            ):  # This can be removed once napari is sped up in the plotting. It changes the shapes only very slightly
                polygons.append(list(df.geometry.iloc[i].exterior.simplify(tolerance=2).coords))
        # this will only work for polygons and not for multipolygons
        polygons = _swap_coordinates(polygons)

        self._viewer.add_shapes(
            polygons,
            name=key,
            affine=affine,
            shape_type="polygon",
            metadata={
                "adata": self._sdata.table[
                    self._sdata.table.obs[self._sdata.table.uns["spatialdata_attrs"]["region_key"]] == key
                ],
                "shapes_key": self._sdata.table.uns["spatialdata_attrs"]["region_key"],
                "shapes_type": "polygons",
            },
        )

    def _add_shapes(self, key: str) -> None:
        if type(self._sdata.shapes[key].iloc[0][0]) == shapely.geometry.point.Point:
            self._add_circles(key)
        elif (type(self._sdata.shapes[key].iloc[0][0]) == shapely.geometry.polygon.Polygon) or (
            type(self._sdata.shapes[key].iloc[0][0]) == shapely.geometry.multipolygon.MultiPolygon
        ):
            self._add_polygons(key)
        else:
            raise TypeError(
                "Incorrect data type passed for shapes (should be Shapely Point or Polygon or MultiPolygon)."
            )

    def _add_label(self, key: str) -> None:
        affine = _get_transform(self._sdata.labels[key], self.coordinate_system_widget._system)

        self._viewer.add_labels(
            self._sdata.labels[key],
            name=key,
            affine=affine,
            metadata={
                "adata": self._sdata.table[
                    self._sdata.table.obs[self._sdata.table.uns["spatialdata_attrs"]["region_key"]] == key
                ],
                "labels_key": self._sdata.table.uns["spatialdata_attrs"]["instance_key"],
            },
        )

    def _add_image(self, key: str) -> None:
        img = self._sdata.images[key]
        affine = _get_transform(self._sdata.images[key], self.coordinate_system_widget._system)

        if isinstance(img, MultiscaleSpatialImage):
            img = img["scale0"][key]
        # TODO: type check
        self._viewer.add_image(
            img,
            name=key,
            affine=affine,
        )

    def _add_points(self, key: str) -> None:
        points = self._sdata.points[key].compute()
        affine = _get_transform(self._sdata.points[key], self.coordinate_system_widget._system)
        if len(points) < 100000:
            subsample = np.arange(len(points))
        else:
            logger.info("Subsampling points because the number of points exceeds the currently supported 100 000.")
            gen = np.random.default_rng()
            subsample = gen.choice(len(points), size=100000, replace=False)

        self._viewer.add_points(
            points[["y", "x"]].values[subsample],
            name=key,
            size=20,
            affine=affine,
            metadata={
                "adata": AnnData(obs=points.loc[subsample, :], obsm={"spatial": points[["x", "y"]].values[subsample]}),
            },
        )


class Interactive:
    """
    Interactive visualization of spatial data.

    Parameters
    ----------
    sdata
        SpatialData object.

    Returns
    -------
    None
    """

    def __init__(self, sdata: SpatialData):
        self._viewer = napari.Viewer()
        self._sdata = sdata
        self._sdata_widget = SdataWidget(self._viewer, sdata)
        self._list_widget = self._viewer.window.add_dock_widget(
            self._sdata_widget, name="SpatialData", area="left", menu=self._viewer.window.window_menu
        )

    def run(self) -> None:
        napari.run()

    def screenshot(self) -> NDArrayA | Any:
        return self._viewer.screenshot(canvas_only=False)
