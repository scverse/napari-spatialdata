from __future__ import annotations
from typing import TypeVar, Any, TYPE_CHECKING, Optional
import napari
from loguru import logger
from napari_spatialdata._utils import save_fig, NDArrayA
from napari_spatialdata._view import QtAdataViewWidget
from anndata import AnnData
import numpy as np
import itertools

# # cannot import these because of cyclic dependencies with spatialdata
# SpatialData = TypeVar("SpatialData")
# BaseElement = TypeVar("BaseElement")

if TYPE_CHECKING:
    from spatialdata import SpatialData
    from spatialdata._core.elements import BaseElement, Image, Labels, Points, Polygons

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

    def __init__(self, sdata: SpatialData):
        self._viewer = napari.Viewer()
        self._add_layers_from_sdata(sdata=sdata)
        self._adata_view = QtAdataViewWidget(viewer=self._viewer)
        napari.run()

    def add_spatial_element(
        self, element: BaseElement, name: Optional[str] = None, annotation_table: Optional[AnnData] = None
    ) -> None:
        from spatialdata._core.elements import Image, Labels, Points, Polygons

        if isinstance(element, Image):
            # ignoring the annotation table
            self._add_image(element, name=name)
        elif isinstance(element, Labels):
            self._add_labels(element, annotation_table=annotation_table, name=name)
        elif isinstance(element, Points):
            self._add_points(element, annotation_table=annotation_table, name=name)
        elif isinstance(element, Polygons):
            self._add_polygons(element, annotation_table=annotation_table, name=name)
        else:
            raise ValueError(f"Unsupported element type: {type(element)}")

    def _add_image(self, image: Image, name: Optional[str] = None) -> None:
        scale = image.transforms.scale_factors
        translate = image.transforms.translation
        self._viewer.add_image(image.data.transpose(), rgb=False, name=name, scale=scale, translate=translate)
        print("TODO: correct transform")

    def _add_labels(self, labels: Labels, annotation_table: Optional[AnnData] = None, name: Optional[str] = None) -> None:
        pass

    def _add_points(self, points: Points, annotation_table: Optional[AnnData] = None, name: Optional[str] = None) -> \
            None:
        adata = points.data
        spatial = adata.obsm["spatial"]
        if "region_radius" in adata.obsm:
            radii = adata.obsm["region_radius"]
        else:
            radii = 1
        annotation = self._find_annotation_for_points(points=points, annotation_table=annotation_table)
        if annotation is not None:
            metadata = {"adata": annotation}
        else:
            metadata = None
        self._viewer.add_points(
            spatial, name=name, edge_color="white", face_color="white", size=radii, metadata=metadata
        )
        # img1, rgb=True, name="image1", metadata={"adata": adata, "library_id": "V1_Adult_Mouse_Brain"}, scale=(1, 1)

    def _find_annotation_for_points(self, points: Points, annotation_table: Optional[AnnData] = None) -> Optional[
        AnnData]:
        """Find the annotation for a points layer from the annotation table."""
        return None

    def _add_polygons(self, polygons: Polygons, annotation_table: Optional[AnnData] = None, name: Optional[str] = None) -> None:
        pass

    def _add_layers_from_sdata(self, sdata: SpatialData):
        ##
        merged = itertools.chain.from_iterable(
            (sdata.images.items(), sdata.labels.items(), sdata.points.items(), sdata.polygons.items())
        )
        for name, element in merged:
            self.add_spatial_element(element, annotation_table=sdata.table, name=name)
        ##
        # viewer.add_image(
        #     img1,
        #     rgb=True,
        #     name="image1",
        # )
        # viewer.add_labels(
        #     label1,
        #     name="label1",
        #     metadata={
        #         "adata": adata1,
        #         "library_id": "1",
        #         "labels_key": "cell_ID",
        #         "points": points1,
        #         "point_diameter": 10,
        #     },  # adata in labels layer will color segments
        # )
        # viewer.add_image(
        #     img2,
        #     rgb=True,
        #     name="image2",
        # )
        # viewer.add_labels(
        #     label2,
        #     name="label2",
        #     metadata={
        #         "adata": adata2,
        #         "library_id": "2",
        #         "labels_key": "cell_ID",
        #         "points": points2,
        #         "point_diameter": 10,
        #     },  # adata in labels layer will color segments
        # )
        ##

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
