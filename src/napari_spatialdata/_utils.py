from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from dask.dataframe.core import DataFrame as DaskDataFrame
from datatree import DataTree
from geopandas import GeoDataFrame
from loguru import logger
from matplotlib.colors import is_color_like, to_rgb
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from numba import njit, prange
from pandas.api.types import infer_dtype, is_categorical_dtype
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)
from scipy.sparse import issparse, spmatrix
from scipy.spatial import KDTree
from shapely import MultiPolygon, Point, Polygon
from skimage.measure import regionprops
from spatial_image import SpatialImage
from spatialdata import SpatialData
from spatialdata.models import SpatialElement, get_axes_names
from spatialdata.transformations import get_transformation

from napari_spatialdata._categoricals_utils import (
    add_colors_for_categorical_sample_annotation,
)
from napari_spatialdata._constants._pkg_constants import Key

if TYPE_CHECKING:
    from xarray import DataArray
try:
    from numpy.typing import NDArray

    NDArrayA = NDArray[Any]
except (ImportError, TypeError):
    NDArray = np.ndarray  # type: ignore[misc]
    NDArrayA = np.ndarray  # type: ignore[misc]


Vector_name_t = Tuple[Optional[Union[pd.Series, NDArrayA]], Optional[str]]


def _ensure_dense_vector(fn: Callable[..., Vector_name_t]) -> Callable[..., Vector_name_t]:
    @wraps(fn)
    def decorator(self: Any, *args: Any, **kwargs: Any) -> Vector_name_t:
        normalize = kwargs.pop("normalize", False)
        res, fmt = fn(self, *args, **kwargs)
        if res is None:
            return None, None

        if isinstance(res, pd.Series):
            if is_categorical_dtype(res):
                return res, fmt
            if is_string_dtype(res) or is_object_dtype(res) or is_bool_dtype(res):
                return res.astype("category"), fmt
            if is_integer_dtype(res):
                unique = res.unique()
                n_uniq = len(unique)
                if n_uniq <= 2 and (set(unique) & {0, 1}):
                    return res.astype(bool).astype("category"), fmt
                if len(unique) <= len(res) // 100:
                    return res.astype("category"), fmt
            elif not is_numeric_dtype(res):
                raise TypeError(f"Unable to process `pandas.Series` of type `{infer_dtype(res)}`.")
            res = res.to_numpy()
        elif issparse(res):
            if TYPE_CHECKING:
                assert isinstance(res, spmatrix)
            res = res.toarray()
        elif not isinstance(res, (np.ndarray, Sequence)):
            raise TypeError(f"Unable to process result of type `{type(res).__name__}`.")

        res = np.asarray(np.squeeze(res))
        if res.ndim != 1:
            raise ValueError(f"Expected 1-dimensional array, found `{res.ndim}`.")

        return (_min_max_norm(res) if normalize else res), fmt

    return decorator


def _set_palette(
    adata: AnnData,
    key: str,
    palette: Optional[str] = None,
    vec: Optional[pd.Series] = None,
) -> dict[Any, Any]:
    if vec is not None and not is_categorical_dtype(vec):
        raise TypeError(f"Expected a `categorical` type, found `{infer_dtype(vec)}`.")

    add_colors_for_categorical_sample_annotation(
        adata,
        key=key,
        vec=vec,
        force_update_colors=palette is not None,
        palette=palette,  # type: ignore[arg-type]
    )
    vec = vec if vec is not None else adata.obs[key]

    return dict(zip(vec.cat.categories, [to_rgb(i) for i in adata.uns[Key.uns.colors(key)]]))


def _get_categorical(
    adata: AnnData,
    key: str,
    vec: Optional[pd.Series] = None,
    palette: Optional[str] = None,
    colordict: Union[pd.Series, dict[Any, Any], None] = None,
) -> NDArrayA:
    categorical = vec if vec is not None else adata.obs[key]
    if not isinstance(colordict, dict):
        col_dict = _set_palette(adata, key, palette, colordict)
    else:
        col_dict = colordict
        for cat in colordict:
            if cat not in categorical.cat.categories:
                raise ValueError(
                    f"The key `{cat}` in the given dictionary is not an existing category in anndata[`{key}`]."
                )
            elif not is_color_like(colordict[cat]):  # noqa: RET506
                raise ValueError(f"`{colordict[cat]}` is not an acceptable color.")

    logger.debug(f"KEY: {key}")
    return np.array([col_dict[v] for v in categorical])


def _position_cluster_labels(coords: NDArrayA, clusters: pd.Series) -> dict[str, NDArrayA]:
    if not is_categorical_dtype(clusters):
        raise TypeError(f"Expected `clusters` to be `categorical`, found `{infer_dtype(clusters)}`.")
    coords = coords[:, 1:]
    df = pd.DataFrame(coords)
    df["clusters"] = clusters.values
    df = df.groupby("clusters")[[0, 1]].apply(lambda g: list(np.median(g.values, axis=0)))
    df = pd.DataFrame(list(df), index=df.index).dropna()
    kdtree = KDTree(coords)
    clusters = np.full(len(coords), fill_value="", dtype=object)
    # index consists of the categories that need not be string
    clusters[kdtree.query(df.values)[1]] = df.index.astype(str)
    return {"clusters": clusters}


def _min_max_norm(vec: Union[spmatrix, NDArrayA]) -> NDArrayA:
    if issparse(vec):
        if TYPE_CHECKING:
            assert isinstance(vec, spmatrix)
        vec = vec.toarray().squeeze()
    vec = np.asarray(vec, dtype=np.float64)
    if vec.ndim != 1:
        raise ValueError(f"Expected `1` dimension, found `{vec.ndim}`.")

    maxx, minn = np.nanmax(vec), np.nanmin(vec)

    return (  # type: ignore[no-any-return]
        np.ones_like(vec) if np.isclose(minn, maxx) else ((vec - minn) / (maxx - minn))
    )


def _get_transform(element: SpatialElement, coordinate_system_name: Optional[str] = None) -> NDArrayA:
    affine: NDArrayA
    transformations = get_transformation(element, get_all=True)
    cs = transformations.keys().__iter__().__next__() if coordinate_system_name is None else coordinate_system_name
    ct = transformations[cs]
    affine = ct.to_affine_matrix(input_axes=("y", "x"), output_axes=("y", "x"))

    if not isinstance(element, (SpatialImage, MultiscaleSpatialImage, AnnData, DaskDataFrame, GeoDataFrame)):
        raise RuntimeError("Cannot get transform for {type(element)}")

    return affine


@njit(cache=True, fastmath=True)
def _point_inside_triangles(triangles: NDArrayA) -> np.bool_:
    # modified from napari
    AB = triangles[:, 1, :] - triangles[:, 0, :]
    AC = triangles[:, 2, :] - triangles[:, 0, :]
    BC = triangles[:, 2, :] - triangles[:, 1, :]

    s_AB = -AB[:, 0] * triangles[:, 0, 1] + AB[:, 1] * triangles[:, 0, 0] >= 0
    s_AC = -AC[:, 0] * triangles[:, 0, 1] + AC[:, 1] * triangles[:, 0, 0] >= 0
    s_BC = -BC[:, 0] * triangles[:, 1, 1] + BC[:, 1] * triangles[:, 1, 0] >= 0

    return np.any((s_AB != s_AC) & (s_AB == s_BC))


@njit(parallel=True)
def _points_inside_triangles(points: NDArrayA, triangles: NDArrayA) -> NDArrayA:
    out = np.empty(
        len(
            points,
        ),
        dtype=np.bool_,
    )
    for i in prange(len(out)):
        out[i] = _point_inside_triangles(triangles - points[i])

    return out


def _transform_to_rgb(element: Union[SpatialImage, MultiscaleSpatialImage]) -> Tuple[DataArray, bool]:
    """Swap the axes to y, x, c if an image supports rgb(a) visualization.

    Checks whether c dim is present in the axes and allows for rgb(a) visualization. If so, subsequently transposes it
    into c x y x x and flags as suitable for rgb visualization.

    Parameters
    ----------
    element: Union[SpatialImage, MultiScaleSpatialImage]
        Element in sdata.images

    Returns
    -------
    new_raster: DataArray
        The image in shape of c x y x x.
    rgb: bool
        Flag indicating suitability for rgb(a) visualization
    """
    axes = get_axes_names(element)

    if "c" in axes:
        assert axes.index("c") == 0
        if isinstance(element, SpatialImage):
            n_channels = element.shape[0]
        elif isinstance(element, MultiscaleSpatialImage):
            v = element["scale0"].values()
            assert len(v) == 1
            n_channels = v.__iter__().__next__().shape[0]
        else:
            raise TypeError(f"Unsupported type for images or labels: {type(element)}")
    else:
        n_channels = 0

    if n_channels in [3, 4]:
        rgb = True
        new_raster = element.transpose("y", "x", "c")
    else:
        rgb = False
        new_raster = element

    # TODO: after we call .transpose() on a MultiscaleSpatialImage object we get a DataTree object. We should look at
    # this better and either cast somehow back to MultiscaleSpatialImage or fix/report this
    if isinstance(new_raster, (MultiscaleSpatialImage, DataTree)):
        list_of_xdata = []
        for k in new_raster:
            v = new_raster[k].values()
            assert len(v) == 1
            xdata = v.__iter__().__next__()
            list_of_xdata.append(xdata)
        new_raster = list_of_xdata

    return new_raster, rgb


def _init_colors_for_obs(adata: AnnData) -> AnnData:
    from scanpy.plotting._utils import _set_colors_for_categorical_obs

    # quick and dirty to set the colors for all the categorical dtype so that if we subset the table the
    # colors are consistent

    for key in adata.obs:
        if is_categorical_dtype(adata.obs[key]) and f"{key}_colors" not in adata.uns:
            _set_colors_for_categorical_obs(adata, key, palette="tab20")
    return adata


def points_to_anndata(points_element: DaskDataFrame, points: NDArrayA, dims: tuple[str]) -> Optional[AnnData]:
    annotations_columns = list(set(points_element.columns.to_list()).difference(dims))
    if len(annotations_columns) > 0:
        df = points_element[annotations_columns].compute()
        annotation = AnnData(shape=(len(points), 0), obs=df, obsm={"spatial": points})
        return _init_colors_for_obs(annotation)
    return None


def _get_mapping_info(annotation_table: AnnData) -> tuple[str, str, str]:
    regions = annotation_table.uns["spatialdata_attrs"]["region"]
    regions_key = annotation_table.uns["spatialdata_attrs"]["region_key"]
    instance_key = annotation_table.uns["spatialdata_attrs"]["instance_key"]
    return regions, regions_key, instance_key


def _find_annotation_for_labels(
    labels: Union[SpatialImage, MultiscaleSpatialImage],
    element_path: str,
    annotating_rows: AnnData,
    instance_key: str,
) -> AnnData:
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
    merged = pd.merge(annotating_rows.obs, centroids, left_on=instance_key, right_on="v", how="left", indicator=True)
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
    circles: GeoDataFrame, element_path: str, annotating_rows: AnnData, instance_key: str
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
            raise ValueError(
                "Partial annotation is support only when the annotation table contains instances not in circles."
            )

    # this is to reorder the circles to match the order of the annotation table
    a = annotating_rows.obs[instance_key].to_numpy()
    b = circles.index.to_numpy()
    sorter = np.argsort(a)
    mapper = sorter[np.searchsorted(a, b, sorter=sorter)]
    assert np.all(a[mapper] == b)
    annotating_rows = annotating_rows[mapper].copy()

    dims = get_axes_names(circles)
    assert dims == ("x", "y") or dims == ("x", "y", "z")
    if type(circles.geometry.iloc[0]) == Point:
        columns = [circles.geometry.x, circles.geometry.y]
        radii = circles["radius"].to_numpy()
    elif type(circles.geometry.iloc[0]) == Polygon:
        columns = [circles.geometry.centroid.x, circles.geometry.centroid.y]
        radii = np.sqrt(circles.geometry.area / np.pi).to_numpy()
    else:
        raise NotImplementedError(f"Unsupported geometry type: {type(circles.geometry.iloc[0])}")

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
    annotating_rows.obsm["region_radius"] = radii
    return annotating_rows


def _find_annotation_for_regions(
    base_element: SpatialElement, element_path: str, annotation_table: Optional[AnnData] = None
) -> Optional[AnnData]:
    if annotation_table is None:
        return None

    regions, regions_key, instance_key = _get_mapping_info(annotation_table)
    if element_path in regions:
        assert regions is not None
        annotating_rows = annotation_table[annotation_table.obs[regions_key] == element_path, :]
        if len(annotating_rows) == 0:
            logger.warning(f"Layer {element_path} expected to be annotated but no annotation found")
            return None

        if isinstance(base_element, (SpatialImage, MultiscaleSpatialImage)):
            table = _find_annotation_for_labels(
                labels=base_element,
                element_path=element_path,
                annotating_rows=annotating_rows,
                instance_key=instance_key,
            )
        elif isinstance(base_element, GeoDataFrame):
            shape = base_element.geometry.iloc[0]
            if isinstance(shape, (Polygon, MultiPolygon, Point)):
                table = _find_annotation_for_shapes(
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

    return None


def _get_ellipses_from_circles(centroids: NDArrayA, radii: NDArrayA) -> NDArrayA:
    """Convert circles to ellipses.

    Parameters
    ----------
    centroids
        Centroids of the circles.
    radii
        Radii of the circles.

    Returns
    -------
    NDArrayA
        Ellipses.
    """
    ndim = centroids.shape[1]
    assert ndim == 2
    r = np.stack([radii] * ndim, axis=1)
    lower_left = centroids - r
    upper_right = centroids + r
    r[:, 0] = -r[:, 0]
    lower_right = centroids - r
    upper_left = centroids + r
    return np.stack([lower_left, lower_right, upper_right, upper_left], axis=1)


def get_metadata_mapping(
    sdata: SpatialData,
    element: SpatialElement,
    coordinate_systems: list[str],
    annotation: Optional[AnnData] = None,
) -> dict[str, Union[SpatialData, SpatialElement, str, Optional[AnnData]]]:
    metadata = {"adata": annotation} if annotation is not None else {}
    metadata["sdata"] = sdata
    metadata["element"] = element
    metadata["coordinate_systems"] = coordinate_systems
    return metadata
