# you can also use the cli, e.g.:
# python -m napari_spatialdata view spatialdata-sandbox/merfish/data.zarr

import spatialdata as sd
import click
from napari_spatialdata import Interactive
import pathlib
import os

DATASETS = ["merfish", "nanostring_cosmx", "mibitof", "toy", "visium", "visium2", "xenium"]


@click.command()
@click.argument("dataset")
def view(dataset):
    """
    Visiualize spatial data from the spatialdata-sandbox datasets repository. Available repositories are merfish,
    nanostring_cosmx, mibitof, toy, visium.
    """
    # use symlinks to make the spatialdata-sandbox datasets repository available
    parent_folder = pathlib.Path(__file__).parent.parent.parent.resolve()
    path = parent_folder / "spatialdata-sandbox"
    if not os.path.isdir(path):
        raise FileNotFoundError(
            'Please add a symlink (or clone) the "spatialdata-sandbox" repository into the parent '
            "folder of this repository."
        )

    path = path / dataset / "data.zarr"
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Dataset {dataset} not found in the file storage.")

    # sdata = sd.SpatialData.read(path, filter_table=True)
    sdata = sd.SpatialData.read(path)
    print(sdata)

    interactive = Interactive(sdata=sdata)


if __name__ == "__main__":
    view()
