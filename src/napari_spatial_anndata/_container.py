from copy import copy, deepcopy
from typing import Any, Union, Optional
from pathlib import Path

import xarray as xr

from src.napari_spatial_anndata._utils import NDArrayA
from src.napari_spatial_anndata._constants._pkg_constants import Key

Pathlike_t = Union[str, Path]
Arraylike_t = Union[NDArrayA, xr.DataArray]
Input_t = Union[Pathlike_t, Arraylike_t, "Container"]


class Container:
    """
    Container for in memory arrays or on-disk images.

    Wraps :class:`xarray.Dataset` to store several image layers with the same `x`, `y` and `z` dimensions in one object.
    Dimensions of stored images are ``(y, x, z, channels)``. The channel dimension may vary between image layers.

    This class also allows for lazy loading and processing using :mod:`dask`, and is given to all image
    processing functions, along with :class:`anndata.AnnData` instance, if necessary.

    Parameters
    ----------
    %(add_img.parameters)s
    scale
        Scaling factor of the image with respect to the spatial coordinates
        saved in the accompanying :class:`anndata.AnnData`.

    Raises
    ------
    %(add_img.raises)s
    """

    def __init__(
        self,
        img: Optional[Input_t] = None,
        layer: str = "image",
        lazy: bool = True,
        scale: float = 1.0,
        **kwargs: Any,
    ):
        self._data: xr.Dataset = xr.Dataset()
        self._data.attrs[Key.img.scale] = scale

    @classmethod
    def _from_dataset(cls, data: xr.Dataset, deep: bool | None = None) -> "Container":
        """
        Utility function used for initialization.

        Parameters
        ----------
        data
            The :class:`xarray.Dataset` to use.
        deep
            If `None`, don't copy the ``data``. If `True`, make a deep copy of the data, otherwise, make a shallow copy.

        Returns
        -------
        The newly created container.
        """  # noqa: D401
        res = cls()
        res._data = data if deep is None else data.copy(deep=deep)
        res._data.attrs.setdefault(Key.img.scale, 1.0)
        return res

    def save(self, path: Pathlike_t, **kwargs: Any) -> None:
        """
        Save the container into a *Zarr* store.

        Parameters
        ----------
        path
            Path to a *Zarr* store.

        Returns
        -------
        Nothing, just saves the container.
        """
        attrs = self.data.attrs
        try:
            self._data = self.data.load()  # if we're loading lazily and immediately saving
            self.data.attrs = {k: v for k, v in self.data.attrs.items()}
            self.data.to_zarr(str(path), mode="w", **kwargs, **kwargs)
        finally:
            self.data.attrs = attrs

    @property
    def data(self) -> xr.Dataset:
        """Underlying :class:`xarray.Dataset`."""
        return self._data

    @property
    def shape(self) -> tuple[int, int]:
        """Image shape ``(y, x)``."""
        return self.data.dims["y"], self.data.dims["x"]

    def copy(self, deep: bool = False) -> "Container":
        """
        Return a copy of self.

        Parameters
        ----------
        deep
            Whether to make a deep copy or not.

        Returns
        -------
        Copy of self.
        """
        return deepcopy(self) if deep else copy(self)
