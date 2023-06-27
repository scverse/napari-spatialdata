from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

import napari
import numpy as np
import shapely
from anndata import AnnData
from loguru import logger
from multiscale_spatial_image import MultiscaleSpatialImage
from napari.viewer import Viewer
from qtpy.QtWidgets import QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget
from spatialdata import SpatialData

from napari_spatialdata.utils._utils import NDArrayA, _get_transform, _swap_coordinates, _transform_to_rgb

if TYPE_CHECKING:
    from napari.utils.events.event import Event


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
        self.coordinate_system_widget.itemClicked.connect(self._update_layers_visibility)

    def _onClick(self, text: str) -> None:
        if self.elements_widget._elements[text] == "labels":
            self._add_label(text)
        elif self.elements_widget._elements[text] == "images":
            self._add_image(text)
        elif self.elements_widget._elements[text] == "points":
            self._add_points(text)
        elif self.elements_widget._elements[text] == "shapes":
            self._add_shapes(text)

    def _update_visible_in_coordinate_system(self, event: Event) -> None:
        """Toggle active in the coordinate system metadata when changing visibility of layer."""
        layer = event.source
        layer_active = layer.metadata["_active_in_cs"]
        selected_coordinate_system = self.coordinate_system_widget._system

        elements = self.elements_widget._elements
        if layer.name in elements:
            if selected_coordinate_system not in layer_active:
                layer_active.add(selected_coordinate_system)
            else:
                layer_active.remove(selected_coordinate_system)

    def _update_layers_visibility(self) -> None:
        """Toggle layer visibility dependent on presence in currently selected coordinate system."""
        elements = self.elements_widget._elements
        coordinate_system = self.coordinate_system_widget._system

        # No layer selected on first time coordinate system selection
        if self._viewer.layers:
            for layer in self._viewer.layers:
                if layer.name not in elements:
                    layer.visible = False
                elif layer.metadata["_active_in_cs"]:
                    layer.visible = True
                    # Prevent _update_visible_in_coordinate_system of invalid removal of coordinate system
                    layer.metadata["_active_in_cs"].add(coordinate_system)

    def _add_circles(self, key: str) -> None:
        selected_cs = self.coordinate_system_widget._system
        df = self._sdata.shapes[key]
        affine = _get_transform(self._sdata.shapes[key], self.coordinate_system_widget._system)
        radii = df.radius.values

        xy = np.array([df.geometry.x, df.geometry.y]).T
        xy = np.fliplr(xy)
        radii = np.array([df.radius[i] for i in range(0, len(df))])

        layer = self._viewer.add_points(
            xy,
            name=key,
            affine=affine,
            size=2 * radii,
            edge_width=0.0,
            metadata={
                "adata": self._sdata.table[
                    self._sdata.table.obs[self._sdata.table.uns["spatialdata_attrs"]["region_key"]] == key
                ],
                "shapes_key": self._sdata.table.uns["spatialdata_attrs"]["region_key"],
                "_active_in_cs": {selected_cs},
            },
        )
        layer.events.visible.connect(self._update_visible_in_coordinate_system)

        # # too slow for medium/large datasets
        # circles = []
        # for i in range(0, len(df)):
        #     circles.append([df.geometry[i].coords[0], [df.radius[i], df.radius[i]]])
        #
        # circles = _swap_coordinates(circles)
        #
        # self._viewer.add_shapes(
        #     circles,
        #     name=key,
        #     affine=affine,
        #     shape_type="ellipse",
        #     metadata={
        #         "adata": self._sdata.table[
        #             self._sdata.table.obs[self._sdata.table.uns["spatialdata_attrs"]["region_key"]] == key
        #         ],
        #         "shapes_key": self._sdata.table.uns["spatialdata_attrs"]["region_key"],
        #         "shapes_type": "circles",
        #     },
        # )

    def _add_polygons(self, key: str) -> None:
        selected_cs = self.coordinate_system_widget._system
        polygons = []
        df = self._sdata.shapes[key]
        affine = _get_transform(self._sdata.shapes[key], selected_cs)

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

        layer = self._viewer.add_shapes(
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
                "name": key,
                "_active_in_cs": {selected_cs},
            },
        )
        layer.events.visible.connect(self._update_visible_in_coordinate_system)

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
        selected_cs = self.coordinate_system_widget._system
        affine = _get_transform(self._sdata.labels[key], selected_cs)

        rgb_labels, _ = _transform_to_rgb(element=self._sdata.labels[key])

        layer = self._viewer.add_labels(
            rgb_labels,
            name=key,
            affine=affine,
            metadata={
                "adata": self._sdata.table[
                    self._sdata.table.obs[self._sdata.table.uns["spatialdata_attrs"]["region_key"]] == key
                ],
                "labels_key": self._sdata.table.uns["spatialdata_attrs"]["instance_key"],
                "name": key,
                "_active_in_cs": {selected_cs},
            },
        )
        layer.events.visible.connect(self._update_visible_in_coordinate_system)

    def _add_image(self, key: str) -> None:
        selected_cs = self.coordinate_system_widget._system
        img = self._sdata.images[key]
        affine = _get_transform(self._sdata.images[key], selected_cs)
        rgb_image, rgb = _transform_to_rgb(element=self._sdata.images[key])

        if isinstance(img, MultiscaleSpatialImage):
            rgb_image[0]
        # TODO: type check
        layer = self._viewer.add_image(
            rgb_image, rgb=rgb, name=key, affine=affine, metadata={"_active_in_cs": {selected_cs}}
        )
        layer.events.visible.connect(self._update_visible_in_coordinate_system)

    def _add_points(self, key: str) -> None:
        selected_cs = self.coordinate_system_widget._system
        points = self._sdata.points[key].compute()
        affine = _get_transform(self._sdata.points[key], selected_cs)
        if len(points) < 100000:
            subsample = np.arange(len(points))
        else:
            logger.info("Subsampling points because the number of points exceeds the currently supported 100 000.")
            gen = np.random.default_rng()
            subsample = gen.choice(len(points), size=100000, replace=False)

        layer = self._viewer.add_points(
            points[["y", "x"]].values[subsample],
            name=key,
            size=20,
            affine=affine,
            edge_width=0.0,
            metadata={
                "adata": AnnData(obs=points.loc[subsample, :], obsm={"spatial": points[["x", "y"]].values[subsample]}),
                "_active_in_cs": {selected_cs},
                "name": key,
            },
        )
        layer.events.visible.connect(self._update_visible_in_coordinate_system)


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
        # will be None when using e.g. python -m napari_spatialdata view
        # will be a napari.Viewer when using napari --plugin napari-spatialdata
        viewer = napari.current_viewer()
        self._viewer = viewer if viewer else napari.Viewer()
        self._sdata_widget = SdataWidget(self._viewer, sdata)
        self._list_widget = self._viewer.window.add_dock_widget(
            self._sdata_widget, name="SpatialData", area="left", menu=self._viewer.window.window_menu
        )

    def run(self) -> None:
        napari.run()

    def screenshot(self) -> NDArrayA | Any:
        return self._viewer.screenshot(canvas_only=False)
