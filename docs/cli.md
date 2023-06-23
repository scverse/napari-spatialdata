# Using the CLI

The Command Line Interface can be accessed by running `python -m napari_spatialdata` or `napari` in a shell.
For extra information, run the help command:

```
python -m napari_spatialdata --help
```

## Opening a dataset

```
python -m napari_spatialdata view <path_to_dataset>
```

The napari `--plugin` flag also allows you to open a dataset. The end result is the same.

```
napari --plugin napari-spatialdata <path_to_dataset>
```
