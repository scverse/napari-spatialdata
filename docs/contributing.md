# Contributing guide

Please refer to the [contribution guide from the `spatialdata` repository](https://github.com/scverse/spatialdata/blob/main/docs/contributing.md).

## Debugging napari GUI tests

To visually inspect what a test is rendering in napari:

1. Change `make_napari_viewer()` to `make_napari_viewer(show=True)`
2. Add `napari.run()` before the end of the test (before the assertions)

Example:

```python
import napari


def test_my_visualization(make_napari_viewer):
    viewer = make_napari_viewer(show=True)
    # ... setup code ...
    napari.run()
    # assertions...
```

Remember to revert these changes before committing.
