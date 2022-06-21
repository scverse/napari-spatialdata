from typing import Any, Tuple, Union, Optional
from dataclasses import field, dataclass

from anndata import AnnData
from napari.layers import Layer
from napari.utils.events import Event, EmitterGroup
import numpy as np
import pandas as pd

from napari_spatialdata._utils import NDArrayA, _ensure_dense_vector
from napari_spatialdata._constants._constants import Symbol
from napari_spatialdata._constants._pkg_constants import Key

__all__ = ["ImageModel"]


@dataclass
class ImageModel:
    """Model which holds the data for interactive visualization."""

    events: EmitterGroup = field(init=False, default=None, repr=True)
    _layer: Layer = field(init=False, default=None, repr=True)
    _adata: AnnData = field(init=False, default=None, repr=True)
    _adata_layer: Optional[str] = field(init=False, default=None, repr=False)
    _spatial_key: str = field(default=Key.obsm.spatial, repr=False)
    _label_key: Optional[str] = field(default=None, repr=True)
    _coordinates: Optional[NDArrayA] = field(init=False, default=None, repr=True)
    _scale: Optional[float] = field(init=False, default=None)

    _spot_diameter: Union[NDArrayA, float] = field(init=False, default=1)
    _scale_key: Optional[str] = field(init=False, default="tissue_hires_scalef")  # TODO(giovp): use constants for these

    palette: Optional[str] = field(init=False, default=None, repr=False)
    cmap: str = field(init=False, default="viridis", repr=False)
    symbol: str = field(init=False, default=Symbol.DISC, repr=False)

    def __post_init__(self) -> None:
        self.events = EmitterGroup(
            source=self,
            layer=Event,
            adata=Event,
        )

    @property
    def layer(self) -> Optional[Layer]:  # noqa: D102
        return self._layer

    @layer.setter
    def layer(self, layer: Optional[Layer]) -> None:
        self._layer = layer
        self.events.layer()

    @property
    def adata(self) -> AnnData:  # noqa: D102
        return self._adata

    @adata.setter
    def adata(self, adata: AnnData) -> None:
        self._adata = adata
        self.events.adata()

    @property
    def adata_layer(self) -> Optional[str]:  # noqa: D102
        return self._adata_layer

    @adata_layer.setter
    def adata_layer(self, adata_layer: str) -> None:
        self._adata_layer = adata_layer

    @property
    def coordinates(self) -> NDArrayA:  # noqa: D102
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates: NDArrayA) -> None:
        self._coordinates = coordinates

    @property
    def scale(self) -> Optional[float]:  # noqa: D102
        return self._scale

    @scale.setter
    def scale(self, scale: float) -> None:
        self._scale = scale

    @property
    def spot_diameter(self) -> NDArrayA:  # noqa: D102
        return self._spot_diameter

    @spot_diameter.setter
    def spot_diameter(self, spot_diameter: NDArrayA) -> None:
        self._spot_diameter = spot_diameter

    @property
    def labels_key(self) -> Optional[str]:  # noqa: D102
        return self._labels_key

    @labels_key.setter
    def labels_key(self, labels_key: str) -> None:
        self._labels_key = labels_key

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
        return self.adata.obs[name], self._format_key(name)

    @_ensure_dense_vector
    def get_var(self, name: Union[str, int], **_: Any) -> Tuple[Optional[NDArrayA], str]:  # TODO(giovp): fix docstring
        """
        Return a gene.

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

    def get_items(self, attr: str) -> Tuple[str, ...]:
        """
        Return valid keys for an attribute.

        Parameters
        ----------
        attr
            Attribute of :mod:`anndata.AnnData` to access.

        Returns
        -------
        The available items.
        """
        adata = self.adata
        if attr in ("obs", "obsm"):
            return tuple(map(str, getattr(adata, attr).keys()))
        return tuple(map(str, getattr(adata, attr).index))

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
