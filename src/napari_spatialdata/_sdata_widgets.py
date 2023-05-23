from __future__ import annotations

from typing import Iterable, Union

import numpy as np
import shapely
from loguru import logger
from napari.viewer import Viewer
from qtpy.QtWidgets import QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget
from spatialdata import SpatialData
from spatialdata.models import Image3DModel, get_axes_names, get_model
from spatialdata.transformations import get_transformation

from napari_spatialdata._utils import _get_transform, _swap_coordinates, _transform_to_rgb, points_to_anndata


class ElementWidget(QListWidget):
    def __init__(self, sdata: SpatialData):
        super().__init__()
        self._sdata = sdata

    def _onClickChange(self, selected_coordinate_system: Union[QListWidgetItem, int, Iterable[str]]) -> None:
        """Change list of elements displayed when selected coordinate system has changed."""
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

    def _select_coord_sys(self, selected_coordinate_system: Union[QListWidgetItem, int, Iterable[str]]) -> None:
        self._system = str(selected_coordinate_system)


class SdataWidget(QWidget):
    def __init__(self, viewer: Viewer, sdata: SpatialData, images: bool, labels: bool, shapes: bool, points: bool):
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
            },
        )

    def _add_polygons(self, key: str) -> None:
        polygons = []
        df = self._sdata.shapes[key]
        affine = _get_transform(self._sdata.shapes[key], self.coordinate_system_widget._system)

        for i in range(0, len(df)):
            polygons.append(list(df.geometry[i].exterior.coords))

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
            },
        )

    def _add_shapes(self, key: str) -> None:
        if type(self._sdata.shapes[key].iloc[0][0]) == shapely.geometry.point.Point:
            self._add_circles(key)
        elif type(self._sdata.shapes[key].iloc[0][0]) == shapely.geometry.polygon.Polygon:
            self._add_polygons(key)
        else:
            raise TypeError("Incorrect data type passed for shapes (should be Shapely Point or Polygon).")

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
        img_element = self._sdata.images[key]
        coordinate_systems = list(get_transformation(img_element, get_all=True).keys())
        if get_model(img_element) == Image3DModel:
            logger.warning("3D images are not supported yet. Skipping.")
            return

        affine = _get_transform(self._sdata.images[key], self.coordinate_system_widget._system)
        new_image, rgb = _transform_to_rgb(element=img_element)

        metadata = {"coordinate_systems": coordinate_systems, "sdata": self._sdata, "element": img_element}
        self._viewer.add_image(
            new_image,
            rgb=rgb,
            name=key,
            affine=affine,
            visible=True,
            metadata=metadata,
        )

    def _add_points(self, key: str) -> None:
        points_element = self._sdata.points[key]
        dims = get_axes_names(points_element)
        point_coords = points_element[list(dims)].compute().values

        MAX_POINTS = 100000
        affine = _get_transform(self._sdata.points[key], self.coordinate_system_widget._system)

        if len(point_coords) < MAX_POINTS:
            choice = None
        else:
            logger.warning(
                f"Too many points {len(point_coords)} > {MAX_POINTS}, subsampling to {MAX_POINTS}. "
                f"Performance will be improved in a future PR"
            )
            gen = np.random.default_rng()
            choice = gen.choice(len(point_coords), size=MAX_POINTS, replace=False)
            point_coords = point_coords[choice, :]

        annotation = points_to_anndata(points_element, point_coords, dims, choice)

        metadata = {"adata": annotation, "library_id": key} if annotation is not None else {}

        coordinate_systems = list(get_transformation(points_element, get_all=True).keys())
        metadata["coordinate_systems"] = coordinate_systems
        metadata["sdata"] = self._sdata
        metadata["element"] = points_element

        if "z" in dims:
            assert len(dims) == 3
            point_coords = point_coords[:, :2]

        radii = 1
        self._viewer.add_points(
            point_coords,
            name=key,
            ndim=2,
            face_color="white",
            size=2 * radii,
            metadata=metadata,
            edge_width=0.0,
            affine=affine,
            visible=True,
        )
