from __future__ import annotations

from copy import copy, deepcopy
from types import MappingProxyType
from typing import (
    Any,
    Union,
    Mapping,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    TYPE_CHECKING,
)
from pathlib import Path
from functools import singledispatchmethod

from scanpy import logging as logg
from typing_extensions import Literal
import numpy as np
import xarray as xr
import dask.array as da

from napari_spatial_anndata._io import (
    _lazy_load_image,
    _infer_dimensions,
    _assert_dims_present,
)
from napari_spatial_anndata._utils import NDArrayA
from napari_spatial_anndata._coords import (
    CropCoords,
    CropPadding,
    _NULL_COORDS,
    _NULL_PADDING,
)
from napari_spatial_anndata._constants._constants import InferDimensions
from napari_spatial_anndata._constants._pkg_constants import Key

Pathlike_t = Union[str, Path]
Arraylike_t = Union[NDArrayA, xr.DataArray]
Input_t = Union[Pathlike_t, Arraylike_t, "Container"]
Pathlike_t = Union[str, Path]
InferDims_t = Union[Literal["default", "prefer_channels", "prefer_z"], Sequence[str]]


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

        if img is not None:
            self.add_img(img, layer=layer, **kwargs)
            if not lazy:
                self.compute()

    @classmethod
    def load(cls, path: Pathlike_t, lazy: bool = True, chunks: Optional[int] = None) -> Container:
        """
        Load data from a *Zarr* store.

        Parameters
        ----------
        path
            Path to *Zarr* store.
        lazy
            Whether to use :mod:`dask` to lazily load image.
        chunks
            Chunk size for :mod:`dask`. Only used when ``lazy = True``.

        Returns
        -------
        The loaded container.
        """
        res = cls()
        res.add_img(path, layer="image", chunks=chunks, lazy=True)

        return res if lazy else res.compute()

    def add_img(
        self,
        img: Input_t,
        layer: Optional[str] = None,
        dims: InferDims_t = InferDimensions.DEFAULT.s,
        library_id: Optional[Union[str, Sequence[str]]] = None,
        lazy: bool = True,
        chunks: Optional[Union[str, tuple[int, ...]]] = None,
        copy: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Add a new image to the container.

        Parameters
        ----------
        img
            In-memory 2, 3 or 4-dimensional array or a path to an on-disk image.
        %(img_layer)s
        dims
            Where to save channel dimension when reading from a file or loading an array. Valid options are:

                - `{id.CHANNELS_LAST.s!r}` - load the last non-spatial dimension as channels.
                - `{id.Z_LAST.s!r}` - load the last non-spatial dimension as Z-dimension.
                - `{id.DEFAULT.s!r}` - same as `{id.CHANNELS_LAST.s!r}`, but for 4-dimensional arrays,
                  tries to also load the first dimension as channels if the last non-spatial dimension is 1.
                - a sequence of dimension names matching the shape of ``img``, e.g. ``('y', 'x', 'z', 'channels')``.
                  `'y'`, `'x'` and `'z'` must always be present.

        library_id
            Name for each Z-dimension of the image. This should correspond to the ``library_id``
            in :attr:`anndata.AnnData.uns`.
        lazy
            Whether to use :mod:`dask` to lazily load image.
        chunks
            Chunk size for :mod:`dask`. Only used when ``lazy = True``.
        copy
            Whether to copy the underlying data if ``img`` is an in-memory array.

        Returns
        -------
        Nothing, just adds a new ``layer`` to :attr:`data`.

        Raises
        ------
        ValueError
            If loading from a file/store with an unknown format or if a supplied channel dimension cannot be aligned.
        NotImplementedError
            If loading a specific data type has not been implemented.
        """
        layer = self._get_next_image_id("image") if layer is None else layer
        dims: InferDimensions | Sequence[str] = (  # type: ignore[no-redef]
            InferDimensions(dims) if isinstance(dims, str) else dims
        )
        res: xr.DataArray | None = self._load_img(img, chunks=chunks, layer=layer, copy=copy, dims=dims, **kwargs)

        if res is not None:
            library_id = self._get_library_ids(library_id, res, allow_new=not len(self))
            try:
                res = res.assign_coords({"z": library_id})
            except ValueError as e:
                if "conflicting sizes for dimension 'z'" not in str(e):
                    raise
                # at this point, we know the container is not empty
                raise ValueError(
                    f"Expected image to have `{len(self.library_ids)}` Z-dimension(s), found `{res.sizes['z']}`."
                ) from None

            if TYPE_CHECKING:
                assert isinstance(res, xr.DataArray)
            logg.info(f"{'Overwriting' if layer in self else 'Adding'} image layer `{layer}`")
            try:
                self.data[layer] = res
            except ValueError as e:
                c_dim = res.dims[-1]
                if f"along dimension {str(c_dim)!r} cannot be aligned" not in str(e):
                    raise
                channel_dim = self._get_next_channel_id(res)
                logg.warning(f"Channel dimension cannot be aligned with an existing one, using `{channel_dim}`")

                self.data[layer] = res.rename({res.dims[-1]: channel_dim})

            if not lazy:
                self.compute(layer)

    @singledispatchmethod
    def _load_img(self, img: Pathlike_t | Input_t | Container, layer: str, **kwargs: Any) -> xr.DataArray | None:
        if isinstance(img, Container):
            if layer not in img:
                raise KeyError(f"Image identifier `{layer}` not found in `{img}`.")

            _ = kwargs.pop("dims", None)
            return self._load_img(img[layer], **kwargs)

        raise NotImplementedError(f"Loading `{type(img).__name__}` is not yet implemented.")

    @_load_img.register(str)
    @_load_img.register(Path)
    def _(
        self,
        img: Pathlike_t,
        chunks: int | None = None,
        dims: InferDimensions | tuple[str, ...] = InferDimensions.DEFAULT,
        **_: Any,
    ) -> xr.DataArray | None:
        def transform_metadata(data: xr.Dataset) -> xr.Dataset:
            for img in data.values():
                _assert_dims_present(img.dims)

            data.attrs[Key.img.coords] = CropCoords.from_tuple(data.attrs.get(Key.img.coords, _NULL_COORDS.to_tuple()))
            data.attrs[Key.img.padding] = CropPadding.from_tuple(
                data.attrs.get(Key.img.padding, _NULL_PADDING.to_tuple())
            )
            data.attrs.setdefault(Key.img.mask_circle, False)
            data.attrs.setdefault(Key.img.scale, 1)

            return data

        img = Path(img)
        logg.debug(f"Loading data from `{img}`")

        if not img.exists():
            raise OSError(f"Path `{img}` does not exist.")

        suffix = img.suffix.lower()

        if suffix in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            return _lazy_load_image(img, dims=dims, chunks=chunks)

        if img.is_dir():
            if len(self._data):
                raise ValueError("Loading data from `Zarr` store is disallowed when the container is not empty.")

            self._data = transform_metadata(xr.open_zarr(str(img), chunks=chunks))
        elif suffix in (".nc", ".cdf"):
            if len(self._data):
                raise ValueError("Loading data from `NetCDF` is disallowed when the container is not empty.")

            self._data = transform_metadata(xr.open_dataset(img, chunks=chunks))
        else:
            raise ValueError(f"Unknown suffix `{img.suffix}`.")

    @_load_img.register(da.Array)
    @_load_img.register(np.ndarray)
    def _(
        self,
        img: NDArrayA,
        copy: bool = True,
        dims: InferDimensions | tuple[str, ...] = InferDimensions.DEFAULT,
        **_: Any,
    ) -> xr.DataArray:
        logg.debug(f"Loading `numpy.array` of shape `{img.shape}`")

        return self._load_img(xr.DataArray(img), copy=copy, dims=dims, warn=False)

    @_load_img.register(xr.DataArray)
    def _(
        self,
        img: xr.DataArray,
        copy: bool = True,
        warn: bool = True,
        dims: InferDimensions | tuple[str, ...] = InferDimensions.DEFAULT,
        **_: Any,
    ) -> xr.DataArray:
        logg.debug(f"Loading `xarray.DataArray` of shape `{img.shape}`")

        img = img.copy() if copy else img
        if not ("y" in img.dims and "x" in img.dims and "z" in img.dims):
            _, dims, _, expand_axes = _infer_dimensions(img, infer_dimensions=dims)
            if TYPE_CHECKING:
                assert isinstance(dims, Iterable)
            if warn:
                logg.warning(f"Unable to find `y`, `x` or `z` dimension in `{img.dims}`. Renaming to `{dims}`")
            # `axes` is always of length 0, 1 or 2
            if len(expand_axes):
                dimnames = ("z", "channels") if len(expand_axes) == 2 else (("channels",) if "z" in dims else ("z",))
                img = img.expand_dims([d for _, d in zip(expand_axes, dimnames)], axis=expand_axes)
            img = img.rename(dict(zip(img.dims, dims)))

        return img.transpose("y", "x", "z", ...)

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
