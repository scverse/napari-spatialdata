# >>>>>> this file will be deleted. It contains old squidpy code to port to the new codebase <<<<<<
#
# from __future__ import annotations
#
# from typing import Any, TYPE_CHECKING, TypeVar
# from pathlib import Path
#
# import napari
# from scanpy import logging as logg
# from anndata import AnnData
#
# from pandas.core.dtypes.common import is_categorical_dtype
# import numpy as np
# import pandas as pd
# import xarray as xr
#
# from PyQt5.QtWidgets import QLabel, QWidget, QGridLayout
#
# from napari import Viewer
# from napari.layers import Points, Shapes
#
# from napari_spatialdata._model import ImageModel
# from napari_spatialdata._view import QtAdataViewWidget
# from napari_spatialdata._utils import _display_channelwise, NDArrayA
# from functools import singledispatchmethod
#
# SpatialData = TypeVar("SpatialData")  # cannot import because of cyclic dependencies
#
# __all__ = ["ImageController"]
#
# # label string: attribute name
# _WIDGETS_TO_HIDE = {
#     "symbol:": "symbolComboBox",
#     "point size:": "sizeSlider",
#     "face color:": "faceColorEdit",
#     "edge color:": "edgeColorEdit",
#     "out of slice:": "outOfSliceCheckBox",
# }
#
#
# class ImageController:
#     """
#     Controller class.
#
#     Parameters
#     ----------
#     %(adata)s
#     %(img_container)s
#     """
#
#     def __init__(self, sdata: SpatialData, viewer: napari.Viewer):
#         self._view = QtAdataViewWidget(viewer=viewer)
#
#     def add_image(self, layer: str) -> bool:
#         """
#         Add a new :mod:`napari` image layer.
#
#         Parameters
#         ----------
#         layer
#             Layer in the underlying's :class:`ImageContainer` which contains the image.
#
#         Returns
#         -------
#         `True` if the layer has been added, otherwise `False`.
#         """
#         if layer in self.view.layernames:
#             self._handle_already_present(layer)
#             return False
#
#         if self.model.container.data[layer].attrs.get("segmentation", False):
#             return self.add_labels(layer)
#
#         img: xr.DataArray = self.model.container.data[layer].transpose("z", "y", "x", ...)
#         multiscale = np.prod(img.shape[1:3]) > (2**16) ** 2
#         n_channels = img.shape[-1]
#
#         rgb = img.attrs.get("rgb", None)
#         if n_channels == 1:
#             rgb, colormap = False, "gray"
#         else:
#             colormap = self.model.cmap
#
#         if rgb is None:
#             logg.debug("Automatically determining whether image is an RGB image")
#             rgb = not _display_channelwise(img.data)
#
#         if rgb:
#             contrast_limits = None
#         else:
#             img = img.transpose(..., "z", "y", "x")  # channels first
#             contrast_limits = float(img.min()), float(img.max())
#
#         logg.info(f"Creating image `{layer}` layer")
#         self.view.viewer.add_image(
#             img.data,
#             name=layer,
#             rgb=rgb,
#             colormap=colormap,
#             blending=self.model.blending,
#             multiscale=multiscale,
#             contrast_limits=contrast_limits,
#         )
#
#         return True
#
#     def add_labels(self, layer: str) -> bool:
#         """
#         Add a new :mod:`napari` labels layer.
#
#         Parameters
#         ----------
#         layer
#             Layer in the underlying's :class:`ImageContainer` which contains the labels image.
#
#         Returns
#         -------
#         `True` if the layer has been added, otherwise `False`.
#         """
#         # beware `update_library` in view.py - needs to be in this order
#         img: xr.DataArray = self.model.container.data[layer].transpose(..., "z", "y", "x")
#         if img.ndim != 4:
#             logg.warning(f"Unable to show image of shape `{img.shape}`, too many dimensions")
#             return False
#
#         if img.shape[0] != 1:
#             logg.warning(f"Unable to create labels layer of shape `{img.shape}`, too many channels `{img.shape[0]}`")
#             return False
#
#         if not np.issubdtype(img.dtype, np.integer):
#             # could also return to `add_images` and render it as image
#             logg.warning(f"Expected label image to be a subtype of `numpy.integer`, found `{img.dtype}`")
#             return False
#
#         logg.info(f"Creating label `{layer}` layer")
#         self.view.viewer.add_labels(
#             img.data,
#             name=layer,
#             multiscale=np.prod(img.shape[-2:]) > (2**16) ** 2,
#         )
#
#         return True
#
#     def add_points(self, vec: NDArrayA | pd.Series, layer_name: str, key: str | None = None) -> bool:
#         """
#         Add a new :mod:`napari` points layer.
#
#         Parameters
#         ----------
#         vec
#             Values to plot. If :class:`pandas.Series`, it is expected to be categorical.
#         layer_name
#             Name of the layer to add.
#         key
#             Key from :attr:`anndata.AnnData.obs` from where the data was taken from.
#             Only used when ``vec`` is :class:`pandas.Series`.
#
#         Returns
#         -------
#         `True` if the layer has been added, otherwise `False`.
#         """
#         if layer_name in self.view.layernames:
#             self._handle_already_present(layer_name)
#             return False
#
#         logg.info(f"Creating point `{layer_name}` layer")
#         properties = self._get_points_properties(vec, key=key)
#         layer: Points = self.view.viewer.add_points(
#             self.model.coordinates,
#             name=layer_name,
#             size=self.model.spot_diameter,
#             opacity=1,
#             blending=self.model.blending,
#             face_colormap=self.model.cmap,
#             edge_colormap=self.model.cmap,
#             symbol=self.model.symbol.v,
#             **properties,
#         )
#         # TODO(michalk8): add contrasting fg/bg color once https://github.com/napari/napari/issues/2019 is done
#         self._hide_points_controls(layer, is_categorical=is_categorical_dtype(vec))
#         layer.editable = False
#
#         return True
