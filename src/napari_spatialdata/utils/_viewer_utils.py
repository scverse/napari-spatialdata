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


def _get_polygons_properties(
    df: GeoDataFrame, simplify: bool, include_z: bool
) -> (tuple)[list[list[tuple[float, float]]], list[int]]:
    # for the moment this function assumes that there are no "Polygon Z", but that the z
    # coordinates, if present, is in a separate column
    indices = []
    polygons = []
    axes = get_axes_names(df)
    include_z = include_z and "z" in axes

    for i in range(0, len(df)):
        indices.append(df.iloc[i].name)
        if include_z:
            z = df.iloc[i].z.item()
        if simplify:
            # This can be removed once napari is sped up in the plotting. It changes the shapes only very slightly
            xy = list(df.geometry.iloc[i].exterior.simplify(tolerance=2).coords)
        else:
            xy = list(df.geometry.iloc[i].exterior.coords)
        coords = xy if not include_z else add_z_to_list_of_xy_tuples(xy=xy, z=z)
        polygons.append(coords)

    return polygons, indices
