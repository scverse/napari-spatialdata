from __future__ import annotations

from typing import Iterable, Union

import numpy as np
from geopandas import GeoDataFrame
from loguru import logger
from napari.viewer import Viewer
from qtpy.QtWidgets import QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget
from shapely import MultiPolygon, Point, Polygon
from spatialdata import SpatialData
from spatialdata.models import Image3DModel, get_axes_names, get_model
from spatialdata.transformations import get_transformation

from napari_spatialdata._utils import (
    _find_annotation_for_regions,
    _get_ellipses_from_circles,
    _get_mapping_info,
    _get_transform,
    _swap_coordinates,
    _transform_to_rgb,
    get_metadata_mapping,
    points_to_anndata,
)


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
        gpdf = self._sdata.shapes[key]
        affine = _get_transform(self._sdata.shapes[key], self.coordinate_system_widget._system)
        coordinate_systems = list(get_transformation(gpdf, get_all=True).keys())
        coords_yx = np.stack([gpdf.geometry.y.to_numpy(), gpdf.geometry.x.to_numpy()], axis=1)

        dims = get_axes_names(gpdf)
        if "z" in dims:
            logger.warning("Circles are currently only supported in 2D. Ignoring z dimension.")

        radii = gpdf.radius.to_numpy() if "radius" in gpdf.columns else 10

        annotation = _find_annotation_for_regions(
            base_element=gpdf, annotation_table=self._sdata.table, element_path=key
        )

        metadata = get_metadata_mapping(self._sdata, gpdf, coordinate_systems, annotation)

        THRESHOLD = 10000
        if len(coords_yx) < THRESHOLD:
            # showing ellipses to overcome https://github.com/scverse/napari-spatialdata/issues/35
            ellipses = _get_ellipses_from_circles(centroids=coords_yx, radii=radii)
            self._viewer.add_shapes(
                ellipses,
                shape_type="ellipse",
                name=key,
                metadata=metadata,
                affine=affine,
            )
        else:
            logger.warning(
                f"Too many shapes {len(coords_yx)} > {THRESHOLD}, using points instead of ellipses. Size will stop "
                f"being correct beyond a certain zoom level"
            )
            # TODO: If https://github.com/scverse/napari-spatialdata/issues/35 is fixed, use points for circles, faster
            self._viewer.add_points(
                coords_yx,
                name=key,
                size=2 * radii,
                metadata=metadata,
                affine=affine,
            )

    def _add_polygons(self, key: str) -> None:
        polygons = []
        gpdf = self._sdata.shapes[key]
        affine = _get_transform(self._sdata.shapes[key], self.coordinate_system_widget._system)

        if "MultiPolygon" in np.unique(gpdf.geometry.type):
            logger.info(
                "Multipolygons are present in the data. For visualization purposes, only the largest polygon per cell "
                "is retained."
            )
            gpdf = gpdf.explode(index_parts=False)
            gpdf["area"] = gpdf.area
            gpdf = gpdf.sort_values(by="area", ascending=False)  # sort by area
            gpdf = gpdf[~gpdf.index.duplicated(keep="first")]  # only keep the largest area
            gpdf = gpdf.sort_index()  # reset the index to the first order

        for shape in gpdf.geometry:
            polygons.append(shape)

        new_shapes = GeoDataFrame(gpdf.drop("geometry", axis=1), geometry=polygons)
        new_shapes.attrs["transform"] = gpdf.attrs["transform"]

        if len(new_shapes) < 100:
            coordinates = new_shapes.geometry.apply(lambda x: np.array(x.exterior.coords).tolist()).tolist()
        else:
            # TODO: This can be removed once napari shape performance improved. Now, changes shapes vis slightly.
            logger.warning("Shape visuals are simplified for performance reasons.")
            coordinates = new_shapes.geometry.apply(
                lambda x: np.array(x.exterior.simplify(tolerance=2).coords).tolist()
            ).tolist()

        annotation = _find_annotation_for_regions(
            base_element=polygons, annotation_table=self._sdata.table, element_path=key
        )
        coordinate_systems = list(get_transformation(gpdf, get_all=True).keys())
        metadata = get_metadata_mapping(self._sdata, polygons, coordinate_systems, annotation)

        MAX_POLYGONS = 500
        if len(coordinates) > MAX_POLYGONS:
            coordinates = coordinates[:MAX_POLYGONS]
            logger.warning(
                f"Too many polygons: {len(coordinates)}. Only the first {MAX_POLYGONS} will be shown.",
                UserWarning,
            )
        # Done because of napary requiring y, x
        coordinates = _swap_coordinates(coordinates)

        self._viewer.add_shapes(
            coordinates,
            name=key,
            affine=affine,
            shape_type="polygon",
            metadata=metadata,
        )

    def _add_shapes(self, key: str) -> None:
        shape_type = type(self._sdata.shapes[key].iloc[0][0])
        if shape_type == Point:
            self._add_circles(key)
        elif shape_type in (Polygon, MultiPolygon):
            self._add_polygons(key)
        else:
            raise TypeError("Incorrect data type passed for shapes (should be Shapely Point or Polygon).")

    def _add_label(self, key: str) -> None:
        label_element = self._sdata.labels[key]
        affine = _get_transform(self._sdata.labels[key], self.coordinate_system_widget._system)

        annotation = _find_annotation_for_regions(
            base_element=label_element, element_path=key, annotation_table=self._sdata.table
        )
        coordinate_systems = list(get_transformation(label_element, get_all=True).keys())
        if annotation is not None:
            _, _, instance_key = _get_mapping_info(annotation)
            metadata = get_metadata_mapping(self._sdata, label_element, coordinate_systems, annotation)
            metadata["labels_key"] = instance_key
        else:
            metadata = get_metadata_mapping(self._sdata, label_element, coordinate_systems)

        rgb_labels, rgb = _transform_to_rgb(element=label_element)
        self._viewer.add_labels(rgb_labels, name=key, metadata=metadata, affine=affine)

    def _add_image(self, key: str) -> None:
        img_element = self._sdata.images[key]
        coordinate_systems = list(get_transformation(img_element, get_all=True).keys())
        if get_model(img_element) == Image3DModel:
            logger.warning("3D images are not supported yet. Skipping.")
            return

        affine = _get_transform(self._sdata.images[key], self.coordinate_system_widget._system)
        new_image, rgb = _transform_to_rgb(element=img_element)
        metadata = get_metadata_mapping(self._sdata, img_element, coordinate_systems)

        self._viewer.add_image(
            new_image,
            rgb=rgb,
            name=key,
            affine=affine,
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
        coordinate_systems = list(get_transformation(points_element, get_all=True).keys())
        metadata = get_metadata_mapping(self._sdata, points_element, coordinate_systems, annotation)

        if "z" in dims:
            assert len(dims) == 3
            point_coords = point_coords[:, :2]

        radii = 1
        self._viewer.add_points(
            point_coords,
            name=key,
            ndim=2,
            size=2 * radii,
            metadata=metadata,
            affine=affine,
        )
