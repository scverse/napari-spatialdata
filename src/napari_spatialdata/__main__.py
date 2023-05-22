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
@click.argument("paths", nargs=-1, required=True)
def view(paths: tuple[str]) -> None:
    """Interactive visualization of SpatialData datasets with napari.

    :param paths: Paths to one or more SpatialData datasets.
    """
    import spatialdata as sd

    from napari_spatialdata import Interactive

    sdatas = []

    # TODO: remove when multiple datasets are supported
    if len(paths) > 1:
        logger.warning("More than one path provided. Only the first path will be used.")

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
        # TODO: remove when multiple datasets are supported
        break

    # TODO: remove [0] when multiple datasets are supported
    interactive = Interactive(sdata=sdatas[0])
    interactive.run()


if __name__ == "__main__":
    main()
