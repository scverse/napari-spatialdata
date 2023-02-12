import click
import os


@click.command(help="Interactive visualization of OME-NGFF spatial data (.zarr) with napari")
@click.argument("path", nargs=-1, type=str)
def view(path):
    assert type(path) == tuple

    import spatialdata as sd
    from napari_spatialdata import Interactive

    sdatas = []
    for p in path:
        if not os.path.isdir(p):
            print(
                f"Error: .zarr storage not found at {p}. Please specify a valid OME-NGFF spatial data (.zarr) file. "
                "Example "
                '"python -m '
                'napari_spatialdata view data.zarr"'
            )
            return
        else:
            sdata = sd.SpatialData.read(p)
            print(sdata)
            sdatas.append(sdata)

    interactive = Interactive(sdata=sdatas)


@click.group()
def cli():
    pass


cli.add_command(view)


def main():
    cli()


if __name__ == "__main__":
    main()
