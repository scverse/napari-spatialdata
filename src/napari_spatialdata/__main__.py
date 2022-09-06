import click
import os


@click.command(help="Interactive visualization of OME-NGFF spatial data (.zarr) with napari")
@click.argument("path", default=False, type=str)
def view(path):
    if not os.path.isdir(path):
        print(
            "Error: .zarr storage not found. Please specify a valid OME-NGFF spatial data (.zarr) file. Example "
            '"python -m '
            'napari_spatialdata view data.zarr"'
        )
    else:
        import spatialdata as sd
        from napari_spatialdata import Interactive

        sdata = sd.SpatialData.read(path)
        print(sdata)

        from napari_spatialdata import Interactive

        interactive = Interactive(sdata=sdata)


@click.group()
def cli():
    pass


cli.add_command(view)


def main():
    cli()


if __name__ == "__main__":
    main()
