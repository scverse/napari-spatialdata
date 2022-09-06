# you can also use the cli, e.g.:
# python -m napari_spatialdata view spatialdata-sandbox/merfish/data.zarr

import spatialdata as sd
import click
from pathlib import Path
from napari_spatialdata import Interactive


@click.command()
def merfish():
    view(path="merfish/data.zarr")


@click.command()
def visium():
    view(path="visium/data.zarr")


@click.group()
def cli():
    pass


cli.add_command(merfish)
cli.add_command(visium)


def view(path: str):
    # use symlinks to make the spatialdata-sandbox datasets repository available
    root = Path("spatialdata-sandbox")
    assert root.exists()

    data_dir = root / path
    sdata = sd.SpatialData.read(data_dir)
    print(sdata)

    interactive = Interactive(sdata=sdata)


if __name__ == "__main__":
    cli()
