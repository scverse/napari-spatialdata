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
@click.argument("path", nargs=-1, type=str, required=True)
@click.option(
    "--headless",
    "-h",
    is_flag=True,
    default=False,
    help="Run napari in headless mode. Used for testing.",
)
def view(path: tuple[str], headless: bool) -> None:
    """
    Interactive visualization of SpatialData datasets with napari.

    Parameters
    ----------
    path
        Path to one or more SpatialData datasets.
    """
    assert type(path) == tuple

    import spatialdata as sd

    from napari_spatialdata import Interactive

    # TODO: remove when multiple datasets are supported
    if len(path) > 1:
        logger.warning("More than one path provided. Only the first path will be used.")

    sdatas = []
    for p_str in path:
        p = Path(p_str).resolve()
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
    Interactive(sdata=sdatas[0], headless=headless)


if __name__ == "__main__":
    main()
