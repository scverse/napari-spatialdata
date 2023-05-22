# Using the CLI

The Command Line Interface can be access by running `spatialdata` or `python -m napari-spatialdata` in a shell.

To get extra information, run using the `--help` flag.

```bash
spatialdata --help
```

## Viewer subcommand

The `viewer` subcommand allows you to open a napari viewer with a spatial image loaded.

```bash
spatialdata viewer <path_to_dataset>
```

This will open a napari viewer with the napari-spatialdata plugin and dataset loaded.
