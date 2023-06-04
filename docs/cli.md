# Using the CLI

The Command Line Interface can be access by running `python -m napari_spatialdata` or `napari` in a shell.
Example:
```
python -m napari_spatialdata view path/to/data.zarr
```

To get extra information, run 

```
python -m napari_spatialdata --help
```

## Viewer subcommand

The `viewer` subcommand of `napari` allows you to open a napari viewer with a spatial image loaded.

```bash
napari viewer <path_to_dataset>
```

This will open a napari viewer with the napari-spatialdata plugin and dataset loaded.
