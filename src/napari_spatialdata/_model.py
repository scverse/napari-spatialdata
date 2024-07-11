from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from napari.layers import Layer
from napari.utils.events import EmitterGroup, Event
from spatialdata.models import get_table_keys

from napari_spatialdata.constants._constants import Symbol
from napari_spatialdata.utils._utils import NDArrayA, _ensure_dense_vector

__all__ = ["DataModel"]


@dataclass
class DataModel:
    """Model which holds the data for interactive visualization."""

    events: EmitterGroup = field(init=False, default=None, repr=True)
    _table_names: Sequence[Optional[str]] = field(default_factory=list, init=False)
    _active_table_name: Optional[str] = field(default=None, init=False, repr=True)
    _layer: Layer = field(init=False, default=None, repr=True)
    _adata: Optional[AnnData] = field(init=False, default=None, repr=True)
    _adata_layer: Optional[str] = field(init=False, default=None, repr=False)
    _region_key: Optional[str] = field(default=None, repr=True)
    _instance_key: Optional[str] = field(default=None, repr=True)
    _color_by: str = field(default="", repr=True, init=False)
    _system_name: Optional[str] = field(default=None, repr=True)

    _scale_key: Optional[str] = field(init=False, default="tissue_hires_scalef")  # TODO(giovp): use constants for these

    _palette: Optional[str] = field(init=False, default=None, repr=False)
    _cmap: str = field(init=False, default="viridis", repr=False)
    _symbol: str = field(init=False, default=Symbol.DISC, repr=False)

    VALID_ATTRIBUTES = ("obs", "var", "obsm", "columns_df")

    def __post_init__(self) -> None:
        self.events = EmitterGroup(
            source=self,
            layer=Event,
            adata=Event,
            color_by=Event,
        )

    def get_items(self, attr: str) -> Optional[Tuple[str, ...]]:
        """
        Return valid keys for an attribute.

        Parameters
        ----------
        attr
            Attribute of :mod:`anndata.AnnData` to access or if columns_df indicates
            the column is part of the SpatialElement dataframe and this will be retrieved.

        Returns
        -------
        The available items coerced to a tuple of strings.
        """
        if attr in ("obs", "obsm"):
            return tuple(map(str, getattr(self.adata, attr).keys()))
        if attr == "columns_df" and self.layer is not None and (df_cols := self.layer.metadata.get("_columns_df")):
            return tuple(map(str, df_cols.columns))
        if attr == "var":
            return tuple(map(str, getattr(self.adata, attr).index))
        return None

    @_ensure_dense_vector
    def get_obs(
        self, name: str, **_: Any
    ) -> Tuple[Optional[Union[pd.Series, NDArrayA]], str]:  # TODO(giovp): fix docstring
        """
        Return an observation.

        Parameters
        ----------
        name
            Key in :attr:`anndata.AnnData.obs` to access.

        Returns
        -------
        The values and the formatted ``name``.
        """
        if name not in self.adata.obs.columns:
            raise KeyError(f"Key `{name}` not found in `adata.obs`.")
        if name != self.instance_key:
            obs_column = self.adata.obs[[self.instance_key, name]]
            obs_column = obs_column.set_index(self.instance_key)[name]
        else:
            obs_column = self.adata.obs[name].copy()
            obs_column.index = self.adata.obs[self.instance_key]
        return obs_column, self._format_key(name)

    @_ensure_dense_vector
    def get_columns_df(self, name: Union[str, int], **_: Any) -> Tuple[Optional[NDArrayA], str]:
        """
        Return a column of the dataframe of the SpatialElement.

        Parameters
        ----------
        name
            Name of the column in the dataframe to retrieve.

        Returns
        -------
        The dataframe column of interest and the formatted name of the column.
        """
        if self.layer is None:
            raise ValueError("Layer must be present")
        return self.layer.metadata["_columns_df"][name], self._format_key(name)

    @_ensure_dense_vector
    def get_var(self, name: Union[str, int], **_: Any) -> Tuple[Optional[NDArrayA], str]:  # TODO(giovp): fix docstring
        """
        Return a column in anndata.var_names.

        Parameters
        ----------
        name
            Gene name in :attr:`anndata.AnnData.var_names` or :attr:`anndata.AnnData.raw.var_names`,
            based on :paramref:`raw`.

        Returns
        -------
        The values and the formatted ``name``.
        """
        try:
            ix = self.adata._normalize_indices((slice(None), name))
        except KeyError:
            raise KeyError(f"Key `{name}` not found in `adata.var_names`.") from None

        return self.adata._get_X(layer=self.adata_layer)[ix], self._format_key(name, adata_layer=True)

    @_ensure_dense_vector
    def get_obsm(self, name: str, index: Union[int, str] = 0) -> Tuple[Optional[NDArrayA], str]:
        """
        Return a vector from :attr:`anndata.AnnData.obsm`.

        Parameters
        ----------
        name
            Key in :attr:`anndata.AnnData.obsm`.
        index
            Index of the vector.

        Returns
        -------
        The values and the formatted ``name``.
        """
        if name not in self.adata.obsm:
            raise KeyError(f"Unable to find key `{name!r}` in `adata.obsm`.")
        res = self.adata.obsm[name]
        pretty_name = self._format_key(name, index=index)

        if isinstance(res, pd.DataFrame):
            try:
                if isinstance(index, str):
                    return res[index], pretty_name
                if isinstance(index, int):
                    return res.iloc[:, index], self._format_key(name, index=res.columns[index])
            except KeyError:
                raise KeyError(f"Key `{index}` not found in `adata.obsm[{name!r}].`") from None

        if not isinstance(index, int):
            try:
                index = int(index, base=10)
            except ValueError:
                raise ValueError(
                    f"Unable to convert `{index}` to an integer when accessing `adata.obsm[{name!r}]`."
                ) from None
        res = np.asarray(res)

        return (res if res.ndim == 1 else res[:, index]), pretty_name

    def _format_key(
        self, key: Union[str, int], index: Optional[Union[int, str]] = None, adata_layer: bool = False
    ) -> str:
        if index is not None:
            return str(key) + f":{index}:{self.layer}"
        if adata_layer:
            return str(key) + (f":{self.adata_layer}" if self.adata_layer is not None else ":X") + f":{self.layer}"

        return str(key) + (f":{self.layer}" if self.layer is not None else ":X")

    @property
    def color_by(self) -> str:
        """The name by which the layer is currently colored."""
        return self._color_by

    @color_by.setter
    def color_by(self, color_column: str) -> None:
        self._color_by = color_column
        self.events.color_by()

    @property
    def table_names(self) -> Sequence[Optional[str]]:
        """The table names annotating the current active napari layer, if any."""
        return self._table_names

    @table_names.setter
    def table_names(self, table_names: Sequence[Optional[str]]) -> None:
        self._table_names = table_names

    @property
    def layer(self) -> Optional[Layer]:  # noqa: D102
        """The current active napari layer."""
        return self._layer

    @layer.setter
    def layer(self, layer: Optional[Layer]) -> None:
        self._layer = layer
        self.events.layer()

    @property
    def active_table_name(self) -> Optional[str]:
        """The table name currently active in the widget."""
        return self._active_table_name

    @active_table_name.setter
    def active_table_name(self, active_table_name: Optional[str]) -> None:
        self._active_table_name = active_table_name

    @property
    def adata(self) -> AnnData:  # noqa: D102
        """The anndata object corresponding to the current active table name."""
        return self._adata

    @adata.setter
    def adata(self, adata: AnnData) -> None:
        self._adata = adata
        self.events.adata()

    @property
    def adata_layer(self) -> Optional[str]:  # noqa: D102
        """The current anndata layer."""
        return self._adata_layer

    @adata_layer.setter
    def adata_layer(self, adata_layer: str) -> None:
        self._adata_layer = adata_layer

    @property
    def region_key(self) -> Optional[str]:  # noqa: D102
        """The region key of the currently active table in the widget."""
        if self.adata is not None:
            _, region_key, _ = get_table_keys(self.adata)
            assert isinstance(region_key, str)
            return region_key
        return None

    @property
    def instance_key(self) -> Optional[str]:  # noqa: D102
        """The instance key of the currently active table in the widget."""
        if self.adata is not None:
            _, _, instance_key = get_table_keys(self.adata)
            assert isinstance(instance_key, str)
            return instance_key
        return None

    @property
    def palette(self) -> Optional[str]:  # noqa: D102
        """The palette from which to draw the colors."""
        return self._palette

    @palette.setter
    def palette(self, palette: str) -> None:
        self._palette = palette

    @property
    def cmap(self) -> str:  # noqa: D102
        """The continuous color map to draw colors from."""
        return self._cmap

    @cmap.setter
    def cmap(self, cmap: str) -> None:
        self._cmap = cmap

    @property
    def symbol(self) -> str:  # noqa: D102
        return self._symbol

    @symbol.setter
    def symbol(self, symbol: str) -> None:
        self._symbol = symbol

    @property
    def scale_key(self) -> Optional[str]:  # noqa: D102
        return self._scale_key

    @scale_key.setter
    def scale_key(self, scale_key: str) -> None:
        self._scale_key = scale_key

    @property
    def system_name(self) -> Optional[str]:  # noqa: D102
        """The layer name."""
        return self._system_name

    @system_name.setter
    def system_name(self, system_name: str) -> None:
        self._system_name = system_name
