from __future__ import annotations

from typing import Any, Optional, Union, Tuple
import napari
from loguru import logger
from napari_spatialdata._utils import save_fig, NDArrayA, _get_ellipses_from_circles
import numpy as np
from napari_spatialdata._constants._pkg_constants import Key
from skimage.measure import regionprops
import pandas as pd
from dask.dataframe.core import DataFrame as DaskDataFrame
from xarray import DataArray
from anndata import AnnData
from spatial_image import SpatialImage
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from geopandas import GeoDataFrame
from spatialdata import SpatialData
from spatialdata.transformations import get_transformation
from spatialdata.models import get_axes_names
from spatialdata.models import get_model, Image3DModel, Labels3DModel, SpatialElement
from datatree import DataTree
from pandas.api.types import is_categorical_dtype
from shapely import Polygon, MultiPolygon, Point

import matplotlib.pyplot as plt

__all__ = ["Interactive", "_get_transform"]


def _get_transform(element: SpatialElement, coordinate_system_name: Optional[str] = None) -> np.ndarray:
    affine: np.ndarray
    transformations = get_transformation(element, get_all=True)
    if coordinate_system_name is None:
        cs = transformations.keys().__iter__().__next__()
    else:
        cs = coordinate_system_name
    ct = transformations[cs]
    affine = ct.to_affine_matrix(input_axes=("y", "x"), output_axes=("y", "x"))

    if isinstance(element, SpatialImage) or isinstance(element, MultiscaleSpatialImage):
        return affine
    elif isinstance(element, AnnData) or isinstance(element, DaskDataFrame) or isinstance(element, GeoDataFrame):
        from spatialdata.transformations import Affine, Sequence, MapAxis

        # old code, I just noticed it was wrong by reading it... but things were being displayed correctly. Luck?
        # new_affine = Sequence(
        #     [Affine(affine, input_axes=("y", "x"), output_axes=("y", "x")), MapAxis({"x": "y", "y": "x"})]
        # )
        # this code flips the coordinates of the non-raster data (points, shapes and polygons), without having to
        # actually flipping the data (which is slow), but just by concatenating a mapAxis
        new_affine = Sequence(
            [MapAxis({"x": "y", "y": "x"}), Affine(affine, input_axes=("y", "x"), output_axes=("y", "x"))]
        )
        new_matrix = new_affine.to_affine_matrix(input_axes=("y", "x"), output_axes=("y", "x"))

        return new_matrix
    else:
        raise RuntimeError("Cannot get transform for {type(element)}")


class Interactive:
    """
    Interactive viewer for spatial data.

    Parameters
    ----------
    %(img_container)s
    %(_interactive.parameters)s
    """

    def __init__(
        self,
        sdata: SpatialData | list[SpatialData],
        images: bool = True,
        labels: bool = True,
        shapes: bool = True,
        points: bool = True,
        with_widgets: bool = True,
        headless: bool = False,
    ):
        # os.environ['NAPARI_ASYNC'] = '1'
        # os.environ['NAPARI_OCTREE'] = '1'
        self._viewer = napari.Viewer()
        if isinstance(sdata, SpatialData):
            sdata = [sdata]
        for s in sdata:
            self._add_layers_from_sdata(sdata=s, images=images, labels=labels, shapes=shapes, points=points)
        if with_widgets:
            self.show_widget()
        if not headless:
            napari.run()

    def show_widget(self):
        """Load the widget for interative features exploration."""
        from napari.plugins import plugin_manager, _npe2

        plugin_manager.discover_widgets()
        _npe2.get_widget_contribution("napari-spatialdata")
        self._viewer.window.add_plugin_dock_widget("napari-spatialdata")

    def _get_affine_for_images_labels(
        self, element: Union[SpatialImage, MultiscaleSpatialImage]
    ) -> Tuple[DataArray, np.ndarray, bool]:
        axes = get_axes_names(element)
        affine = _get_transform(element=element)

        if "c" in axes:
            assert axes.index("c") == 0
            if isinstance(element, SpatialImage):
                n_channels = element.shape[0]
            elif isinstance(element, MultiscaleSpatialImage):
                v = element["scale0"].values()
                assert len(v) == 1
                n_channels = v.__iter__().__next__().shape[0]
            else:
                raise TypeError(f"Unsupported type: {type(element)}")
        else:
            n_channels = 0

        if n_channels in [3, 4]:
            rgb = True
            new_raster = element.transpose("y", "x", "c")
        else:
            rgb = False
            new_raster = element

        # TODO: after we call .transpose() on a MultiscaleSpatialImage object we get a DataTree object. We should look at this better and either cast somehow back to MultiscaleSpatialImage or fix/report this
        if isinstance(new_raster, MultiscaleSpatialImage) or isinstance(new_raster, DataTree):
            list_of_xdata = []
            for k in new_raster.keys():
                v = new_raster[k].values()
                assert len(v) == 1
                xdata = v.__iter__().__next__()
                list_of_xdata.append(xdata)
            new_raster = list_of_xdata

        return new_raster, affine, rgb

    def _add_image(
        self, sdata: SpatialData, image: Union[SpatialImage, MultiscaleSpatialImage], element_path: str = None
    ) -> None:
        new_image, affine, rgb = self._get_affine_for_images_labels(element=image)
        coordinate_systems = list(get_transformation(image, get_all=True).keys())
        metadata = {"coordinate_systems": coordinate_systems, "sdata": sdata, "element": image}
        self._viewer.add_image(
            new_image,
            rgb=rgb,
            name=element_path,
            affine=affine,
            visible=False,
            metadata=metadata,
        )

    def _add_labels(
        self,
        sdata: SpatialData,
        labels: Union[SpatialImage, MultiscaleSpatialImage],
        element_path: str = None,
        annotation_table: Optional[AnnData] = None,
    ) -> None:
        annotation = self._find_annotation_for_regions(
            base_element=labels, element_path=element_path, annotation_table=annotation_table
        )
        if annotation is not None:
            _, _, instance_key = self._get_mapping_info(annotation)
            metadata = {
                "adata": annotation,
                "library_id": element_path,
                "labels_key": instance_key,
                # "points": points1,
                # "point_diameter": 10,
            }
        else:
            metadata = {}
        coordinate_systems = list(get_transformation(labels, get_all=True).keys())
        metadata["coordinate_systems"] = coordinate_systems
        metadata["sdata"] = sdata
        metadata["element"] = labels
        new_labels, affine, rgb = self._get_affine_for_images_labels(element=labels)
        self._viewer.add_labels(new_labels, name=element_path, metadata=metadata, affine=affine, visible=False)

    def _add_shapes(
        self, sdata: SpatialData, shapes: GeoDataFrame, element_path: str, annotation_table: Optional[AnnData] = None
    ) -> None:
        shape = shapes.geometry.iloc[0]
        if isinstance(shape, Polygon) or isinstance(shape, MultiPolygon):
            ##
            new_polygons = []
            for shape in shapes.geometry:
                if isinstance(shape, Polygon):
                    new_polygons.append(shape)
                elif isinstance(shape, MultiPolygon):
                    polygon = shape.geoms[0]
                    new_polygons.append(polygon)
                else:
                    raise TypeError(f"Unsupported type: {type(shape)}")
                # TODO: here we are considering only the first polygon of the multipolygon, we need to addess this
                # if len(shape.geoms) != 1:
                #     raise NotImplementedError(f"MultiPolygons with more zero, or more than one polygon are not supported. Number of polygons: {len(shape.geoms)}")
            new_shapes = GeoDataFrame(shapes.drop("geometry", axis=1), geometry=new_polygons)
            new_shapes.attrs["transform"] = shapes.attrs["transform"]
            self._add_polygons(
                sdata=sdata, polygons=new_shapes, element_path=element_path, annotation_table=annotation_table
            )
            ##
        elif isinstance(shape, Point):
            self._add_circles(sdata=sdata, shapes=shapes, element_path=element_path, annotation_table=annotation_table)
        else:
            raise TypeError(f"Unsupported shape type: {type(shape)}")

    def _add_circles(
        self, sdata: SpatialData, shapes: AnnData, element_path: str, annotation_table: Optional[AnnData] = None
    ) -> None:
        dims = get_axes_names(shapes)
        if "z" in dims:
            logger.warning("Circles are currently only supported in 2D. Ignoring z dimension.")
        x = shapes.geometry.x.to_numpy()
        y = shapes.geometry.y.to_numpy()
        spatial = np.stack([x, y], axis=1)
        if "radius" in shapes.columns:
            radii = shapes.radius.to_numpy()
        else:
            radii = 10
        annotation = self._find_annotation_for_regions(
            base_element=shapes, annotation_table=annotation_table, element_path=element_path
        )
        if annotation is not None:
            # # points_annotation is required from the squidpy legagy code, TODO: remove
            # points_annotation = AnnData(X=shapes.data.X)
            # points_annotation.obs['gene'] = annotation.obs
            # metadata = {"adata": annotation, "library_id": element_path, "shapes": points_annotation}
            metadata = {"adata": annotation, "library_id": element_path}
        else:
            metadata = {}
        coordinate_systems = list(get_transformation(shapes, get_all=True).keys())
        metadata["coordinate_systems"] = coordinate_systems
        metadata["sdata"] = sdata
        metadata["element"] = shapes
        affine = _get_transform(element=shapes)
        # THRESHOLD = 1000000
        THRESHOLD = 10000
        if len(spatial) < THRESHOLD:
            # showing ellipses to overcome https://github.com/scverse/napari-spatialdata/issues/35
            # spatial = spatial[: 1000]
            # radii = radii[: 1000]
            ellipses = _get_ellipses_from_circles(centroids=spatial, radii=radii)
            self._viewer.add_shapes(
                ellipses,
                shape_type="ellipse",
                name=element_path,
                edge_color="white",
                face_color="white",
                metadata=metadata,
                edge_width=0.0,
                affine=affine,
                visible=False,
            )
        else:
            logger.warning(
                f"Too many shapes {len(spatial)} > {THRESHOLD}, using points instead of ellipses. Size will stop being correct beyond a certain zoom level"
            )
            # TODO: when https://github.com/scverse/napari-spatialdata/issues/35 is fixed, use points when we detect cirlces, since points are faster
            self._viewer.add_points(
                spatial,
                name=element_path,
                edge_color="white",
                face_color="white",
                size=2 * radii,
                metadata=metadata,
                edge_width=0.0,
                affine=affine,
                visible=False,
                # canvas_size_limits=(2, 100000)  # this doesn't seem to affect the problem with the point size
            )

    def _add_points(self, sdata: SpatialData, points: DaskDataFrame, element_path: str) -> None:
        dims = get_axes_names(points)
        spatial = points[list(dims)].compute().values
        # np.sum(np.isnan(spatial)) / spatial.size
        radii = 1
        annotations_columns = list(set(points.columns.to_list()).difference(dims))
        if len(annotations_columns) > 0:
            df = points[annotations_columns].compute()
            annotation = AnnData(shape=(len(points), 0), obs=df, obsm={"spatial": spatial})
            self._init_colors_for_obs(annotation)
        else:
            annotation = None
        if annotation is not None:
            # # points_annotation is required from the squidpy legagy code, TODO: remove
            # points_annotation = AnnData(X=shapes.data.X)
            # points_annotation.obs['gene'] = annotation.obs
            # metadata = {"adata": annotation, "library_id": element_path, "shapes": points_annotation}
            metadata = {"adata": annotation, "library_id": element_path}
        else:
            metadata = {}
        coordinate_systems = list(get_transformation(points, get_all=True).keys())
        metadata["coordinate_systems"] = coordinate_systems
        metadata["sdata"] = sdata
        metadata["element"] = points

        MAX_POINTS = 100000
        if len(spatial) > MAX_POINTS:
            logger.warning(
                f"Too many points {len(spatial)} > {MAX_POINTS}, subsampling to {MAX_POINTS}. Performance will be improved in a future PR"
            )
            choice = np.random.choice(len(spatial), MAX_POINTS, replace=False)
            spatial = spatial[choice]
            if annotation is not None:
                annotation = annotation[choice]
                metadata["adata"] = annotation

        # 3d not supported at the moment, let's remove the 3d coordinate
        # TODO: support
        # affine is always 2d
        affine = _get_transform(element=points)
        axes = get_axes_names(points)
        if "z" in axes:
            assert len(axes) == 3
            spatial = spatial[:, :2]
        self._viewer.add_points(
            spatial,
            name=element_path,
            ndim=2,  # len(axes),
            # edge_color="white",
            face_color="white",
            size=2 * radii,
            metadata=metadata,
            edge_width=0.0,
            affine=affine,
            visible=False,
        )
        print("debug")
        ##

    def _add_polygons(
        self, sdata: SpatialData, polygons: GeoDataFrame, element_path: str, annotation_table: Optional[AnnData] = None
    ) -> None:
        coordinates = polygons.geometry.apply(lambda x: np.array(x.exterior.coords).tolist()).tolist()
        annotation = self._find_annotation_for_regions(
            base_element=polygons, annotation_table=annotation_table, element_path=element_path
        )
        if annotation is not None:
            metadata = {"adata": annotation, "library_id": element_path}
        else:
            metadata = {}
        ##
        coordinate_systems = list(get_transformation(polygons, get_all=True).keys())
        metadata["coordinate_systems"] = coordinate_systems
        metadata["sdata"] = sdata
        metadata["element"] = polygons
        affine = _get_transform(element=polygons)
        MAX_POLYGONS = 100
        if len(coordinates) > MAX_POLYGONS:
            coordinates = coordinates[:MAX_POLYGONS]
            logger.warning(
                f"Too many polygons: {len(coordinates)}. Only the first {MAX_POLYGONS} will be shown.",
                UserWarning,
            )
        self._viewer.add_shapes(
            coordinates,
            shape_type="polygon",
            name=element_path,
            edge_width=0.5,  # TODO: choose this number based on the size of spatial elements in a smart way,
            # or let the user choose it
            edge_color="green",
            face_color=np.array([0.0, 0, 0.0, 0.0]),
            metadata=metadata,
            affine=affine,
            visible=False,
        )
        ##

    def _find_annotation_for_regions(
        self, base_element: SpatialElement, element_path: str, annotation_table: Optional[AnnData] = None
    ) -> Optional[AnnData]:
        if annotation_table is None:
            return None

        regions, regions_key, instance_key = self._get_mapping_info(annotation_table)
        if element_path in regions:
            assert regions is not None
            annotating_rows = annotation_table[annotation_table.obs[regions_key] == element_path, :]
            if len(annotating_rows) == 0:
                logger.warning(f"Layer {element_path} expected to be annotated but no annotation found")
                return None
            else:
                if isinstance(base_element, SpatialImage) or isinstance(base_element, MultiscaleSpatialImage):
                    table = self._find_annotation_for_labels(
                        labels=base_element,
                        element_path=element_path,
                        annotating_rows=annotating_rows,
                        instance_key=instance_key,
                    )
                elif isinstance(base_element, GeoDataFrame):
                    shape = base_element.geometry.iloc[0]
                    if isinstance(shape, Polygon) or isinstance(shape, MultiPolygon) or isinstance(shape, Point):
                        table = self._find_annotation_for_shapes(
                            circles=base_element,
                            element_path=element_path,
                            annotating_rows=annotating_rows,
                            instance_key=instance_key,
                        )
                    else:
                        raise TypeError(f"Unsupported shape type: {type(shape)}")
                else:
                    raise ValueError(f"Unsupported element type: {type(base_element)}")
                return table

        else:
            return None

    def _get_mapping_info(self, annotation_table: AnnData):
        regions = annotation_table.uns["spatialdata_attrs"]["region"]
        regions_key = annotation_table.uns["spatialdata_attrs"]["region_key"]
        instance_key = annotation_table.uns["spatialdata_attrs"]["instance_key"]
        return regions, regions_key, instance_key

    def _find_annotation_for_labels(
        self,
        labels: Union[SpatialImage, MultiscaleSpatialImage],
        element_path: str,
        annotating_rows: AnnData,
        instance_key: str,
    ):
        # TODO: use xarray apis
        x = np.array(labels.data)
        u = np.unique(x)
        # backgrond = 0 in u
        # adjacent_labels = (len(u) - 1 if backgrond else len(u)) == np.max(u)
        available_u = annotating_rows.obs[instance_key]
        u_not_annotated = set(np.setdiff1d(u, available_u).tolist())
        if 0 in set(u_not_annotated):
            u_not_annotated.remove(0)
        if len(u_not_annotated) > 0:
            logger.warning(f"{len(u_not_annotated)}/{len(u)} labels not annotated: {u_not_annotated}")
        annotating_rows = annotating_rows[annotating_rows.obs[instance_key].isin(u), :].copy()

        # TODO: requirement due to the squidpy legacy code, in the future this will not be needed
        annotating_rows.uns[Key.uns.spatial] = {}
        annotating_rows.uns[Key.uns.spatial][element_path] = {}
        annotating_rows.uns[Key.uns.spatial][element_path][Key.uns.scalefactor_key] = {}
        annotating_rows.uns[Key.uns.spatial][element_path][Key.uns.scalefactor_key]["tissue_hires_scalef"] = 1.0
        list_of_cx = []
        list_of_cy = []
        list_of_v = []
        regions = regionprops(x)
        for props in regions:
            cx, cy = props.centroid
            v = props.label
            list_of_cx.append(cx)
            list_of_cy.append(cy)
            list_of_v.append(v)
        centroids = pd.DataFrame({"cx": list_of_cx, "cy": list_of_cy, "v": list_of_v})
        merged = pd.merge(
            annotating_rows.obs, centroids, left_on=instance_key, right_on="v", how="left", indicator=True
        )
        # background = merged.query('_merge == "left_only"')
        # assert len(background) == 1
        # assert background.loc[background.index[0], instance_key] == 0
        # index_of_background = merged[merged[instance_key] == 0].index[0]
        # merged.loc[index_of_background, "v"] = 0
        merged["v"] = merged["v"].astype(int)

        assert len(annotating_rows) == len(merged)
        assert annotating_rows.obs[instance_key].tolist() == merged["v"].tolist()

        merged_centroids = merged[["cx", "cy"]].to_numpy()
        assert len(merged_centroids) == len(merged)
        annotating_rows.obsm["spatial"] = merged_centroids
        annotating_rows.obsm["region_radius"] = np.array([10.0] * len(merged_centroids))  # arbitrary value
        return annotating_rows

    def _find_annotation_for_shapes(
        self, circles: GeoDataFrame, element_path: str, annotating_rows: AnnData, instance_key: str
    ) -> Optional[AnnData]:
        """Find the annotation for a circles layer from the annotation table."""
        available_instances = circles.index.tolist()
        annotated_instances = annotating_rows.obs[instance_key].tolist()
        assert len(available_instances) == len(set(available_instances)), (
            "Instance keys must be unique. Found " "multiple regions instances with the " "same key."
        )
        if len(annotated_instances) != len(set(annotated_instances)):
            raise ValueError("Instance keys must be unique. Found multiple regions instances with the same key.")
        available_instances = set(available_instances)
        annotated_instances = set(annotated_instances)
        # TODO: maybe add this check back
        # assert annotated_instances.issubset(available_instances), "Annotation table contains instances not in circles."
        if len(annotated_instances) != len(available_instances):
            if annotated_instances.issuperset(available_instances):
                pass
            else:
                raise ValueError("Partial annotation is support only when the annotation table contains instances not in circles.")

        # this is to reorder the circles to match the order of the annotation table
        a = annotating_rows.obs[instance_key].to_numpy()
        b = circles.index.to_numpy()
        sorter = np.argsort(a)
        mapper = sorter[np.searchsorted(a, b, sorter=sorter)]
        assert np.all(a[mapper] == b)
        annotating_rows = annotating_rows[mapper].copy()

        dims = get_axes_names(circles)
        assert dims == ("x", "y") or dims == ("x", "y", "z")
        columns = [circles.geometry.x, circles.geometry.y]
        if "z" in dims:
            columns.append(circles.geometry.z)
        spatial = np.column_stack(columns)
        # TODO: requirement due to the squidpy legacy code, in the future this will not be needed
        annotating_rows.uns[Key.uns.spatial] = {}
        annotating_rows.uns[Key.uns.spatial][element_path] = {}
        annotating_rows.uns[Key.uns.spatial][element_path][Key.uns.scalefactor_key] = {}
        annotating_rows.uns[Key.uns.spatial][element_path][Key.uns.scalefactor_key]["tissue_hires_scalef"] = 1.0
        annotating_rows.obsm["spatial"] = spatial
        # workaround for the legacy code to support different sizes for different circles
        annotating_rows.obsm["region_radius"] = circles["radius"].to_numpy()
        return annotating_rows

    def _find_annotation_for_polygons(
        self, polygons: GeoDataFrame, element_path: str, annotating_rows: AnnData, instance_key: str
    ) -> Optional[AnnData]:
        print("_find_annotation_for_polygons not implemented")
        return None

    @staticmethod
    def _init_colors_for_obs(adata: AnnData):
        from scanpy.plotting._utils import _set_colors_for_categorical_obs

        # quick and dirty to set the colors for all the categorical dtype so that if we subset the table the
        # colors are consistent
        if adata is not None:
            for key in adata.obs.keys():
                if is_categorical_dtype(adata.obs[key]) and f'{key}_colors' not in adata.uns:
                    _set_colors_for_categorical_obs(adata, key, palette="tab20")

    def _add_layers_from_sdata(
        self,
        sdata: SpatialData,
        images: bool = True,
        labels: bool = True,
        shapes: bool = True,
        points: bool = True,
    ):
        for prefix in ["images", "labels", "shapes", "points"]:
            d = sdata.__getattribute__(prefix)
            annotation_table = sdata.table
            self._init_colors_for_obs(annotation_table)

            for name, element in d.items():
                element_path = name
                if prefix == "images":
                    if get_model(element) == Image3DModel:
                        logger.warning("3D images are not supported yet. Skipping.")
                    else:
                        if images:
                            self._add_image(sdata=sdata, image=element, element_path=element_path)
                elif prefix == "points":
                    if points:
                        self._add_points(sdata=sdata, points=element, element_path=element_path)
                elif prefix == "labels":
                    if labels:
                        if get_model(element) == Labels3DModel:
                            logger.warning("3D labels are not supported yet. Skipping.")
                        else:
                            self._add_labels(
                                sdata=sdata, labels=element, annotation_table=annotation_table, element_path=element_path
                            )
                elif prefix == "shapes":
                    if shapes:
                        self._add_shapes(
                            sdata=sdata, shapes=element, annotation_table=annotation_table, element_path=element_path
                        )
                else:
                    raise ValueError(f"Unsupported element type: {type(element)}")

    def screenshot(
        self,
        return_result: bool = False,
        dpi: float | None = 180,
        save: str | None = None,
        canvas_only: bool = True,
        **kwargs: Any,
    ) -> NDArrayA | None:
        """
        Plot a screenshot of the viewer's canvas.

        Parameters
        ----------
        return_result
            If `True`, return the image as an :class:`numpy.uint8`.
        dpi
            Dots per inch.
        save
            Whether to save the plot.
        canvas_only
            Whether to show only the canvas or also the widgets.
        kwargs
            Keyword arguments for :meth:`matplotlib.axes.Axes.imshow`.

        Returns
        -------
        Nothing, if ``return_result = False``, otherwise the image array.
        """
        try:
            arr = np.asarray(self._viewer.screenshot(path=None, canvas_only=canvas_only))
        except RuntimeError as e:
            logger.error(f"Unable to take a screenshot. Reason: {e}")
            return None

        fig, ax = plt.subplots(nrows=1, ncols=1, dpi=dpi)
        fig.tight_layout()

        ax.imshow(arr, **kwargs)
        plt.axis("off")

        if save is not None:
            save_fig(fig, save)

        return arr if return_result else None

    def close(self) -> None:
        """Close the viewer."""
        self._viewer.close()

    # @property
    # def adata(self) -> AnnData:
    #     """Annotated data object."""
    #     # return self._controller._view.model.adata

    # def __repr__(self) -> str:
    # return f"Interactive view of {repr(self._controller.model.container)}"

    # def __str__(self) -> str:
    #     return repr(self)


if __name__ == "__main__":
    from spatialdata import SpatialData

    sdata = SpatialData.read("/Users/macbook/Downloads/xe_rep1_subset_napari_vis_fixed.zarr")
    Interactive(sdata=sdata, images=True, labels=True, shapes=True, points=True)
    # sdata0 = SpatialData.read("../../spatialdata-sandbox/merfish/data.zarr")
    # sdata1 = SpatialData.read("../../spatialdata-sandbox/visium/data.zarr")
    # sdata1 = SpatialData.read(os.path.expanduser("~/temp/merged.zarr"))
    # Interactive(sdata=[sdata0, sdata1])
