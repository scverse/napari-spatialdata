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
    headless
        Whether to run the napari application or to run in headless mode.
    """
    assert type(path) is tuple

    import spatialdata as sd

    from napari_spatialdata import Interactive

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

    Interactive(sdata=sdatas, headless=headless)


if __name__ == "__main__":
    main()
