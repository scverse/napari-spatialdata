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
from spatialdata import SpatialData, get_transform, SpatialElement, get_dims

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
        self._viewer = napari.Viewer(show=not headless)
        self.sdata = sdata
        self._add_layers_from_sdata(sdata=self.sdata)
        # self._adata_view = QtAdataViewWidget(viewer=self._viewer)
        if with_widgets:
            self.show_widget()
        napari.run()

    def show_widget(self):
        """Load the widget for interative features exploration."""
        from napari.plugins import plugin_manager, _npe2

        plugin_manager.discover_widgets()
        _npe2.get_widget_contribution("napari-spatialdata")
        self._viewer.window.add_plugin_dock_widget("napari-spatialdata")

    def _get_transform(self, element: SpatialElement, coordinate_system_name: Optional[str] = None) -> np.ndarray:
        affine: np.ndarray
        # TODO: refactor when multiple coordinate transformations per element are supported
        # if coordinate_system_name is None:
        #     if len(element.coordinate_systems) > 1:
        #         # TODO: An easy workaround is to add one layer per coordinate system, another better method is to
        #         #  change the affine matrix when the coordinate system is changed.
        #         raise ValueError("Only one coordinate system per element is supported at the moment.")
        #     coordinate_system_name, cs = element.coordinate_systems.items().__iter__().__next__()
        # else:
        #     cs = element.coordinate_systems[coordinate_system_name]
        # ct = element.transformations[coordinate_system_name]
        ct = get_transform(element)
        cs = ct.output_coordinate_system
        from spatialdata import Identity, Scale, Translation, Affine, Rotation, Sequence

        if isinstance(ct, Identity):
            assert len(cs.axes_names) == element.ndim
            affine = np.eye(len(cs.axes_names) + 1, element.ndim + 1)
            # affine = np.hstack((affine, np.zeros((len(cs.axes), 1))))
            # affine = np.vstack((affine, np.zeros((1, element.ndim + 1))))
            # affine[-1, -1] = 1
        elif any([isinstance(ct, tt) for tt in [Scale, Translation, Affine, Rotation, Sequence]]):
            affine = ct.to_affine().affine
        else:
            raise TypeError(f"Unsupported transform type: {type(ct)}")
        # axes of the target coordinate space
        axes = cs.axes_names
        return affine, axes

    def _get_affine_for_images_labels(self, element: Union[SpatialImage, MultiscaleSpatialImage]) -> Tuple[DataArray, np.ndarray, bool]:
        # adjust affine transform
        # napari doesn't want channels in the affine transform, I guess also not time
        # this subset of the matrix is possible because ngff 0.4 assumes that the axes are ordered as [t, c, z, y, x]

        # "dims" are the axes of the element in the source coordinate system (implicit coordinate system of the labels
        # or image)
        if isinstance(element, SpatialImage):
            src_axes = element.dims
        else:
            assert isinstance(element, MultiscaleSpatialImage)
            src_axes = element['scale0'].ds.DAPI.dims
        assert list(src_axes) == [ax for ax in ["t", "c", "z", "y", "x"] if ax in src_axes]
        # "axes" are the axes of the element in the target coordinate system
        affine, des_axes = self._get_transform(element=element)

        # target_order_pre = [ax for ax in des_axes if ax in src_axes]
        # target_order_post = [ax for ax in ['t', 'c', 'z', 'y', 'x'] if ax in des_axes]
        #
        # from spatialdata import Affine

        # fix_order_pre = Affine._get_affine_iniection_from_axes(src_axes, target_order_pre)
        # fix_order_post = Affine._get_affine_iniection_from_axes(des_axes, target_order_post)
        # affine = fix_order_post @ affine @ fix_order_pre

        rows_to_keep = list(range(affine.shape[0]))
        if "t" in des_axes:
            rows_to_keep.remove(des_axes.index("t"))
        if "c" in des_axes:
            rows_to_keep.remove(des_axes.index("c"))
        cols_to_keep = list(range(affine.shape[1]))
        if "t" in src_axes:
            cols_to_keep.remove(src_axes.index("t"))
        if "c" in src_axes:
            cols_to_keep.remove(src_axes.index("c"))
        cropped_affine = affine[np.ix_(rows_to_keep, cols_to_keep)]

        # adjust channel ordering
        rgb = False
        if "t" in src_axes:
            # where do we put the time axis?
            raise NotImplementedError("Time dimension not supported yet")
        if "c" in src_axes:
            assert "c" in src_axes
            assert src_axes.index("c") == 0
            new_order = [src_axes[i] for i in range(1, len(src_axes))] + ["c"]
            if isinstance(element, SpatialImage):
                if element.shape[0] > 1:
                    new_raster = element.transpose(*new_order)
                else:
                    new_raster = element
                if element.shape[0] == 3:
                    rgb = True
            else:
                assert isinstance(element, MultiscaleSpatialImage)
                variables = list(element['scale0'].ds.variables)
                assert len(variables) == 1
                var = variables[0]
                new_raster = [element[k].ds[var] for k in element.keys()]
                if element['scale0'].dims['c'] > 1:
                    new_raster = [data.transpose(*new_order) for data in new_raster]
                if element['scale0'].dims['c'] == 3:
                    rgb = True
        else:
            new_raster = element
        # this code is useless
        # if isinstance(new_raster, SpatialImage):
        #     if new_raster.dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        #         maximum = new_raster.max().compute()
        #         if maximum > 255:
        #             new_raster = new_raster.astype(float)
        # else:
        #     assert isinstance(new_raster, list)
        #     if new_raster[0].dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        #         maximum = new_raster[-1].max().compute()
        #         if maximum > 255:
        #             new_raster = [raster.astype(float) for raster in new_raster]
        return new_raster, cropped_affine, rgb

    def _suffix_from_full_name(self, element_path: str):
        return "/".join(element_path.split("/")[2:])

    def _add_image(self, image: Union[SpatialImage, MultiscaleSpatialImage], element_path: str = None) -> None:
        new_image, affine, rgb = self._get_affine_for_images_labels(element=image)
        ct = get_transform(image)
        cs = ct.output_coordinate_system
        metadata = {"coordinate_systems": [cs.name], "sdata": self.sdata}
        self._viewer.add_image(
            new_image, rgb=rgb, name=self._suffix_from_full_name(element_path), affine=affine, visible=False,
            metadata=metadata
        )

    def _add_labels(self, labels: Union[SpatialImage, MultiscaleSpatialImage], element_path: str = None, annotation_table: Optional[AnnData]
    = None) -> \
            None:
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
        ct = get_transform(labels)
        cs = ct.output_coordinate_system
        metadata["coordinate_systems"] = [cs.name]
        metadata["sdata"] = self.sdata
        new_labels, affine, rgb = self._get_affine_for_images_labels(element=labels)
        self._viewer.add_labels(
            new_labels, name=self._suffix_from_full_name(element_path), metadata=metadata, affine=affine, visible=False
        )

    def _get_affine_for_points_polygons(self, element: Union[GeoDataFrame, pa.Table]) -> np.ndarray:
        # the whole function assumes there is no time dimension
        ndim = len(get_dims(element))
        # "src_axes" are the axes of the element in the source coordinate system (implicit coordinate system of the
        # labels
        src_axes = ['x', 'y', 'z'][:ndim]
        assert tuple(src_axes) == get_dims(element)

        assert list(src_axes) == [ax for ax in ["x", "y", "z"] if ax in src_axes]
        # "axes" are the axes of the element in the target coordinate system
        affine, des_axes = self._get_transform(element=element)

        # target_order_pre = [ax for ax in des_axes if ax in src_axes]
        # target_order_post = [ax for ax in ['t', 'c', 'z', 'y', 'x'] if ax in des_axes]
        #
        # from spatialdata import Affine
        #
        # fix_order_pre = Affine._get_affine_iniection_from_axes(src_axes, target_order_pre)
        # fix_order_post = Affine._get_affine_iniection_from_axes(des_axes, target_order_post)
        # affine = fix_order_post @ affine @ fix_order_pre

        out_space_dim = affine.shape[0] - 1
        in_space_dim = affine.shape[1] - 1
        assert in_space_dim == ndim
        if out_space_dim != ndim:
            assert out_space_dim == ndim + 1
            # remove the first row of the affine since it contains only channel information (it should be zero)
            assert np.isclose(np.linalg.norm(affine[0, :]), 0)
            affine = affine[1:, :]
        return affine

    def _add_shapes(self, shapes: AnnData, element_path: str, annotation_table: Optional[AnnData] = None) -> None:
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
        ct = get_transform(shapes)
        cs = ct.output_coordinate_system
        metadata["coordinate_systems"] = [cs.name]
        metadata["sdata"] = self.sdata
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

    def _add_points(self, points: pa.Table, element_path: str) -> None:
        dims = get_dims(points)
        spatial = points.select(dims).to_pandas().to_numpy()
        radii = 1
        annotations_columns = list(set(points.column_names).difference(dims))
        if len(annotations_columns) > 0:
            df = points.select(annotations_columns).to_pandas()
            annotation = AnnData(shape=(len(points), 0), obs=df, obsm={'spatial': spatial})
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
        ct = get_transform(points)
        cs = ct.output_coordinate_system
        metadata["coordinate_systems"] = [cs.name]
        metadata["sdata"] = self.sdata
        affine = self._get_affine_for_points_polygons(element=points)
        # 3d not supported at the moment, let's remove the 3d coordinate
        # TODO: support
        axes = get_dims(points)
        if 'z' in axes:
            assert len(axes) == 3
            spatial = spatial[:, :2]
            # z is column 2 (input space xyz and row 0 (output space zyx)
            # actually I was expecting the output to be still xyz, but this seem to work...
            # TODO: CHECK!
            affine = affine[1:, np.array([0, 1, 3], dtype=int)]
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

    def _add_polygons(self, polygons: GeoDataFrame, element_path: str, annotation_table: Optional[AnnData] = None) -> None:
        coordinates = polygons.geometry.apply(lambda x: np.array(x.exterior.coords).tolist()).tolist()
        annotation = self._find_annotation_for_regions(
            base_element=polygons, annotation_table=annotation_table, element_path=element_path
        )
        if annotation is not None:
            metadata = {"adata": annotation, "library_id": element_path}
        else:
            metadata = {}
        ##
        ct = get_transform(polygons)
        cs = ct.output_coordinate_system
        metadata["coordinate_systems"] = [cs.name]
        metadata["sdata"] = self.sdata
        affine = self._get_affine_for_points_polygons(element=polygons)
        self._viewer.add_shapes(
            coordinates,
            shape_type="polygon",
            name=self._suffix_from_full_name(element_path),
            edge_width=20.0,
            edge_color="green",
            face_color=np.array([0.0, 0, 0.0, 0.0]),
            metadata=metadata,
            affine=affine,
            visible=False,
        )
        ##

    def _find_annotation_for_regions(
        self, base_element: SpatialElement, element_path: str, annotation_table: Optional[AnnData] =
            None
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
                        labels=base_element, element_path=element_path, annotating_rows=annotating_rows,
                        instance_key=instance_key
                    )
                elif isinstance(base_element, AnnData):
                    return self._find_annotation_for_shapes(
                        points=base_element, element_path=element_path, annotating_rows=annotating_rows,
                        instance_key=instance_key
                    )
                elif isinstance(base_element, GeoDataFrame):
                    return self._find_annotation_for_polygons(
                        polygons=base_element, element_path=element_path, annotating_rows=annotating_rows,
                        instance_key=instance_key
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

    def _find_annotation_for_labels(self, labels: Union[SpatialImage, MultiscaleSpatialImage], element_path: str, annotating_rows:
    AnnData, instance_key:
    str):
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
        for prefix in ["images", "labels", "points", "polygons", "shapes"]:
            d = sdata.__getattribute__(prefix)
            annotation_table = sdata.table
            for name, element in d.items():
                element_path = f"/{prefix}/{name}"
                if prefix == 'images':
                    self._add_image(element, element_path=element_path)
                elif prefix == 'points':
                    self._add_points(element, element_path=element_path)
                elif prefix == 'labels':
                    self._add_labels(element, annotation_table=annotation_table, element_path=element_path)
                elif prefix == 'shapes':
                    self._add_shapes(element, annotation_table=annotation_table, element_path=element_path)
                elif prefix == 'polygons':
                    self._add_polygons(element, annotation_table=annotation_table, element_path=element_path)
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

    # sdata = SpatialData.read("spatialdata-sandbox/merfish/data.zarr")
    sdata = SpatialData.read("spatialdata-sandbox/visium/data.zarr")
    Interactive(sdata)
