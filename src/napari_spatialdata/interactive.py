from __future__ import annotations

from typing import Any, TYPE_CHECKING, Optional, Union, Tuple
import napari
from loguru import logger
from napari_spatialdata._utils import save_fig, NDArrayA
import numpy as np
from napari_spatialdata._constants._pkg_constants import Key
from skimage.measure import regionprops
import pandas as pd
import pyarrow as pa
from xarray import DataArray
from anndata import AnnData
from spatial_image import SpatialImage
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from geopandas import GeoDataFrame
from spatialdata import SpatialData, SpatialElement, get_dims
from datatree import DataTree

import matplotlib.pyplot as plt

__all__ = ["Interactive"]


class Interactive:
    """
    Interactive viewer for spatial data.

    Parameters
    ----------
    %(img_container)s
    %(_interactive.parameters)s
    """

    def __init__(self, sdata: SpatialData, with_widgets: bool = True, headless: bool = False):
        # os.environ['NAPARI_ASYNC'] = '1'
        # os.environ['NAPARI_OCTREE'] = '1'
        self._viewer = napari.Viewer()
        # self._adata_view = QtAdataViewWidget(viewer=self._viewer)
        self._add_layers_from_sdata(sdata=sdata)
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

    def _get_transform(self, element: SpatialElement, coordinate_system_name: Optional[str] = None) -> np.ndarray:
        affine: np.ndarray
        transformations = SpatialData.get_all_transformations(element)
        if coordinate_system_name is None:
            if len(transformations) > 1:
                # TODO: An easy workaround is to add one layer per coordinate system, another better method is to
                #  change the affine matrix when the coordinate system is changed.
                raise ValueError("Only one coordinate system per element is supported at the moment.")
            cs = transformations.keys().__iter__().__next__()
        else:
            cs = transformations[coordinate_system_name]
        ct = transformations[cs]
        affine = ct.to_affine_matrix(input_axes=("y", "x"), output_axes=("y", "x"))
        return affine

    def _get_affine_for_images_labels(
        self, element: Union[SpatialImage, MultiscaleSpatialImage]
    ) -> Tuple[DataArray, np.ndarray, bool]:
        axes = get_dims(element)
        affine = self._get_transform(element=element)

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
            variables = list(new_raster["scale0"].ds.variables)
            assert len(variables) == 1
            var = variables[0]
            new_raster = [new_raster[k].ds[var] for k in new_raster.keys()]

        return new_raster, affine, rgb

    def _suffix_from_full_name(self, element_path: str):
        return "/".join(element_path.split("/")[2:])

    def _add_image(
        self, sdata: SpatialData, image: Union[SpatialImage, MultiscaleSpatialImage], element_path: str = None
    ) -> None:
        new_image, affine, rgb = self._get_affine_for_images_labels(element=image)
        coordinate_systems = list(SpatialData.get_all_transformations(image).keys())
        metadata = {"coordinate_systems": coordinate_systems, "sdata": sdata}
        self._viewer.add_image(
            new_image,
            rgb=rgb,
            name=self._suffix_from_full_name(element_path),
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
        coordinate_systems = list(SpatialData.get_all_transformations(labels).keys())
        metadata["coordinate_systems"] = coordinate_systems
        metadata["sdata"] = sdata
        new_labels, affine, rgb = self._get_affine_for_images_labels(element=labels)
        self._viewer.add_labels(
            new_labels, name=self._suffix_from_full_name(element_path), metadata=metadata, affine=affine, visible=False
        )

    def _get_affine_for_points_polygons(self, element: Union[AnnData, GeoDataFrame, pa.Table]) -> np.ndarray:
        from spatialdata._core.transformations import Affine, Sequence, MapAxis

        affine = self._get_transform(element=element)
        new_affine = Sequence(
            [Affine(affine, input_axes=("y", "x"), output_axes=("y", "x")), MapAxis({"x": "y", "y": "x"})]
        )
        new_matrix = new_affine.to_affine_matrix(input_axes=("y", "x"), output_axes=("y", "x"))
        return new_matrix

    def _add_shapes(
        self, sdata: SpatialData, shapes: AnnData, element_path: str, annotation_table: Optional[AnnData] = None
    ) -> None:
        spatial = shapes.obsm["spatial"]
        if "size" in shapes.obs:
            radii = shapes.obs["size"]
        else:
            radii = 1
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
        coordinate_systems = list(SpatialData.get_all_transformations(shapes).keys())
        metadata["coordinate_systems"] = coordinate_systems
        metadata["sdata"] = sdata
        affine = self._get_affine_for_points_polygons(element=shapes)
        self._viewer.add_points(
            spatial,
            name=self._suffix_from_full_name(element_path),
            edge_color="white",
            face_color="white",
            size=2 * radii,
            metadata=metadata,
            edge_width=0.0,
            affine=affine,
            visible=False,
        )

    def _add_points(self, sdata: SpatialData, points: pa.Table, element_path: str) -> None:
        dims = get_dims(points)
        spatial = points.select(dims).to_pandas().to_numpy()
        radii = 1
        annotations_columns = list(set(points.column_names).difference(dims))
        if len(annotations_columns) > 0:
            df = points.select(annotations_columns).to_pandas()
            annotation = AnnData(shape=(len(points), 0), obs=df, obsm={"spatial": spatial})
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
        coordinate_systems = list(SpatialData.get_all_transformations(points).keys())
        metadata["coordinate_systems"] = coordinate_systems
        metadata["sdata"] = sdata
        affine = self._get_affine_for_points_polygons(element=points)
        # 3d not supported at the moment, let's remove the 3d coordinate
        # TODO: support
        axes = get_dims(points)
        if "z" in axes:
            assert len(axes) == 3
            spatial = spatial[:, :2]
            # z is column 2 (input space xyz and row 0 (output space zyx)
            # actually I was expecting the output to be still xyz, but this seem to work...
            # TODO: CHECK!
            affine = affine[1:, np.array([0, 1, 3], dtype=int)]
        ##
        self._viewer.add_points(
            spatial,
            name=self._suffix_from_full_name(element_path),
            ndim=len(axes),
            edge_color="white",
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
        coordinate_systems = list(SpatialData.get_all_transformations(polygons).keys())
        metadata["coordinate_systems"] = coordinate_systems
        metadata["sdata"] = sdata
        affine = self._get_affine_for_points_polygons(element=polygons)
        MAX_POLYGONS = 10000
        if len(coordinates) > MAX_POLYGONS:
            coordinates = coordinates[:MAX_POLYGONS]
            logger.warning(
                f"Too many polygons: {len(coordinates)}. Only the first {MAX_POLYGONS} will be shown.",
                UserWarning,
            )
        self._viewer.add_shapes(
            coordinates,
            shape_type="polygon",
            name=self._suffix_from_full_name(element_path),
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
            if isinstance(regions, list):
                annotating_rows = annotation_table[annotation_table.obs[regions_key] == element_path, :]
            else:
                assert isinstance(regions, str)
                assert regions_key is None
                annotating_rows = annotation_table
            if len(annotating_rows) == 0:
                logger.warning(f"Layer {element_path} expected to be annotated but no annotation found")
                return None
            else:
                if isinstance(base_element, SpatialImage) or isinstance(base_element, MultiscaleSpatialImage):
                    return self._find_annotation_for_labels(
                        labels=base_element,
                        element_path=element_path,
                        annotating_rows=annotating_rows,
                        instance_key=instance_key,
                    )
                elif isinstance(base_element, AnnData):
                    return self._find_annotation_for_shapes(
                        points=base_element,
                        element_path=element_path,
                        annotating_rows=annotating_rows,
                        instance_key=instance_key,
                    )
                elif isinstance(base_element, GeoDataFrame):
                    return self._find_annotation_for_polygons(
                        polygons=base_element,
                        element_path=element_path,
                        annotating_rows=annotating_rows,
                        instance_key=instance_key,
                    )
                else:
                    raise ValueError(f"Unsupported element type: {type(base_element)}")

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
        u_not_annotated = np.setdiff1d(u, available_u)
        if len(u_not_annotated) > 0:
            logger.warning(f"{len(u_not_annotated)}/{len(u)} labels not annotated: {u_not_annotated}")
        annotating_rows = annotating_rows[annotating_rows.obs[instance_key].isin(u), :]

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
        self, points: AnnData, element_path: str, annotating_rows: AnnData, instance_key: str
    ) -> Optional[AnnData]:
        """Find the annotation for a points layer from the annotation table."""
        available_instances = points.obs.index.tolist()
        annotated_instances = annotating_rows.obs[instance_key].tolist()
        assert len(available_instances) == len(set(available_instances)), (
            "Instance keys must be unique. Found " "multiple regions instances with the " "same key."
        )
        assert len(annotated_instances) == len(set(annotated_instances)), (
            "Instance keys must be unique. Found " "multiple regions instances annotations with the same key."
        )
        available_instances = set(available_instances)
        annotated_instances = set(annotated_instances)
        # TODO: maybe add this check back
        # assert annotated_instances.issubset(available_instances), "Annotation table contains instances not in points."
        if len(annotated_instances) != len(available_instances):
            raise ValueError("TODO: support partial annotation")

        # TODO: requirement due to the squidpy legacy code, in the future this will not be needed
        annotating_rows.uns[Key.uns.spatial] = {}
        annotating_rows.uns[Key.uns.spatial][element_path] = {}
        annotating_rows.uns[Key.uns.spatial][element_path][Key.uns.scalefactor_key] = {}
        annotating_rows.uns[Key.uns.spatial][element_path][Key.uns.scalefactor_key]["tissue_hires_scalef"] = 1.0
        annotating_rows.obsm["spatial"] = points.obsm["spatial"]
        # workaround for the legacy code to support different sizes for different points
        annotating_rows.obsm["region_radius"] = points.obs["size"].to_numpy()
        return annotating_rows

    def _find_annotation_for_polygons(
        self, polygons: GeoDataFrame, element_path: str, annotating_rows: AnnData, instance_key: str
    ) -> Optional[AnnData]:
        print("_find_annotation_for_polygons not implemented")
        return None

    def _add_layers_from_sdata(self, sdata: SpatialData):
        for prefix in ["images", "labels", "shapes", "points", "polygons"]:
            d = sdata.__getattribute__(prefix)
            annotation_table = sdata.table
            for name, element in d.items():
                element_path = f"/{prefix}/{name}"
                if prefix == "images":
                    self._add_image(sdata=sdata, image=element, element_path=element_path)
                elif prefix == "points":
                    self._add_points(sdata=sdata, points=element, element_path=element_path)
                elif prefix == "labels":
                    self._add_labels(
                        sdata=sdata, labels=element, annotation_table=annotation_table, element_path=element_path
                    )
                elif prefix == "shapes":
                    self._add_shapes(
                        sdata=sdata, shapes=element, annotation_table=annotation_table, element_path=element_path
                    )
                elif prefix == "polygons":
                    self._add_polygons(
                        sdata=sdata, polygons=element, annotation_table=annotation_table, element_path=element_path
                    )
                    pass
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

    # sdata = SpatialData.read("merfish/data.zarr")
    sdata = SpatialData.read("visium/data.zarr")
    Interactive(sdata)
