from geopandas import GeoDataFrame


def _get_polygons_properties(df: GeoDataFrame, simplify: bool) -> tuple[list[list[tuple[float, float]]], list[int]]:
    indices = []
    polygons = []

    if simplify:
        for i in range(0, len(df)):
            indices.append(df.iloc[i].name)
            # This can be removed once napari is sped up in the plotting. It changes the shapes only very slightly
            polygons.append(list(df.geometry.iloc[i].exterior.simplify(tolerance=2).coords))
    else:
        for i in range(0, len(df)):
            indices.append(df.iloc[i].name)
            polygons.append(list(df.geometry.iloc[i].exterior.coords))

    return polygons, indices
