from geopandas import GeoDataFrame
from spatialdata.models import get_axes_names

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
            xy = list(df.geometry.iloc[i].exterior.simplify(tolerance=2).coords)
        else:
            xy = list(df.geometry.iloc[i].exterior.coords)

        coords: Polygon2D | Polygon3D
        if add_z:
            z_val = float(df.iloc[i].z.item() if hasattr(df.iloc[i].z, "item") else df.iloc[i].z)
            coords = [(x, y, z_val) for x, y in xy]
        else:
            coords = xy

        polygons.append(coords)

    return polygons, indices
