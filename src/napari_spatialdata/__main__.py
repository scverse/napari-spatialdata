from pathlib import Path

import click
from loguru import logger


@click.group()
def cli() -> None:
    """Group of commands for napari-spatialdata CLI."""
    pass


def main() -> None:
    """Run the napari-spatialdata CLI."""
    cli()


@cli.command(help="Interactive visualization of SpatialData datasets with napari")
@click.argument("path", nargs=-1, type=tuple, required=True)
def view(paths: tuple[str]) -> None:
    """Interactive visualization of SpatialData datasets with napari.

    :param path: Path to the SpatialData dataset
    """
    assert type(paths) == tuple

    import spatialdata as sd

    from napari_spatialdata import Interactive

    sdatas = []
    for path in paths:
        p = Path(path).resolve()
        assert p.exists(), f"Error: {p} does not exist"
        logger.info(f"Reading {p}")
        if not p.is_dir():
            logger.error(
                f"Error: .zarr storage not found at {p}. Please specify a valid OME-NGFF spatial data (.zarr) file. "
                "Example "
                '"python -m '
                'napari_spatialdata view data.zarr"'
            )
            return
        sdata = sd.SpatialData.read(p)
        sdatas.append(sdata)

    # TODO: support multiple spatial data
    interactive = Interactive(sdata=sdatas[0])
    interactive.run()


if __name__ == "__main__":
    main()
