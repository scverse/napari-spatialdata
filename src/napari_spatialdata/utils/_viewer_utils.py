from typing import cast

from geopandas import GeoDataFrame
from spatialdata.models import get_axes_names


def add_z_to_list_of_xy_tuples(xy: list[tuple[float, float]], z: float) -> list[tuple[float, float, float]]:
    """
    Add z coordinates to a list of (x, y) tuples.

    Parameters
    ----------
    xy
        List of (x, y) tuples.
    z
        z coordinate to add to each tuple.

    Returns
    -------
    list[tuple[float, float, float]]
        List of (x, y, z) tuples.
    """
    return [(x, y, z) for x, y in xy]


# type aliases, only used in this module
Coord2D = tuple[float, float]
Coord3D = tuple[float, float, float]
Polygon2D = list[Coord2D]
Polygon3D = list[Coord3D]
Polygon = Polygon2D | Polygon3D


def _get_polygons_properties(df: GeoDataFrame, simplify: bool, include_z: bool) -> tuple[list[Polygon], list[int]]:
    # assumes no "Polygon Z": z is in separate column if present
    indices: list[int] = []
    polygons: list[Polygon] = []

    axes = get_axes_names(df)
    add_z = include_z and "z" in axes

    for i in range(len(df)):
        indices.append(int(df.index[i]))

        if simplify:
            xy = cast(list[Coord2D], list(df.geometry.iloc[i].exterior.simplify(tolerance=2).coords))
        else:
            xy = cast(list[Coord2D], list(df.geometry.iloc[i].exterior.coords))

        coords: Polygon2D | Polygon3D
        if add_z:
            z_val = float(df.iloc[i].z.item() if hasattr(df.iloc[i].z, "item") else df.iloc[i].z)
            coords = add_z_to_list_of_xy_tuples(xy=xy, z=z_val)
        else:
            coords = xy

        polygons.append(coords)

    return polygons, indices
