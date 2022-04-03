from __future__ import annotations

from copy import copy, deepcopy
from types import MappingProxyType
from typing import Any, Union, Mapping, Iterable, Iterator, Optional, Sequence
from pathlib import Path

import xarray as xr

from napari_spatial_anndata._utils import NDArrayA
from napari_spatial_anndata._constants._pkg_constants import Key

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
    def _from_dataset(cls, data: xr.Dataset, deep: Optional[bool] = None) -> Container:
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
    def library_ids(self) -> list[str]:
        """Library ids."""
        try:
            return list(map(str, self.data.coords["z"].values))
        except KeyError:
            return []

    @library_ids.setter
    def library_ids(self, library_ids: Union[str, Sequence[str], Mapping[str, str]]) -> None:
        """Set library ids."""
        if isinstance(library_ids, Mapping):
            library_ids = [str(library_ids.get(lid, lid)) for lid in self.library_ids]
        elif isinstance(library_ids, str):
            library_ids = (library_ids,)

        library_ids = list(map(str, library_ids))
        if len(set(library_ids)) != len(library_ids):
            raise ValueError(f"Remapped library ids must be unique, found `{library_ids}`.")
        self._data = self.data.assign_coords({"z": library_ids})

    @property
    def data(self) -> xr.Dataset:
        """Underlying :class:`xarray.Dataset`."""
        return self._data

    @property
    def shape(self) -> tuple[int, int]:
        """Image shape ``(y, x)``."""
        return self.data.dims["y"], self.data.dims["x"]

    def copy(self, deep: bool = False) -> Container:
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

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __iter__(self) -> Iterator[str]:
        yield from self.data.keys()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: str) -> xr.DataArray:
        return self.data[key]

    def _ipython_key_completions_(self) -> Iterable[str]:
        return sorted(map(str, self.data.keys()))

    def __copy__(self) -> Container:
        return type(self)._from_dataset(self.data, deep=False)

    def __deepcopy__(self, memodict: Mapping[str, Any] = MappingProxyType({})) -> Container:
        return type(self)._from_dataset(self.data, deep=True)

    def _repr_html_(self) -> str:
        import html

        if not len(self):
            return f"{self.__class__.__name__} object with 0 layers"

        inflection = "" if len(self) <= 1 else "s"
        s = f"{self.__class__.__name__} object with {len(self.data.keys())} layer{inflection}:"
        style = "text-indent: 25px; margin-top: 0px; margin-bottom: 0px;"

        for i, layer in enumerate(self.data.keys()):
            s += f"<p style={style!r}><strong>{html.escape(str(layer))}</strong>: "
            s += ", ".join(
                f"<em>{html.escape(str(dim))}</em> ({shape})"
                for dim, shape in zip(self.data[layer].dims, self.data[layer].shape)
            )
            s += "</p>"
            if i == 9 and i < len(self) - 1:  # show only first 10 layers
                s += f"<p style={style!r}>and {len(self) - i  - 1} more...</p>"
                break

        return s

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}[shape={self.shape}, layers={sorted(self.data.keys())}]"

    def __str__(self) -> str:
        return repr(self)
