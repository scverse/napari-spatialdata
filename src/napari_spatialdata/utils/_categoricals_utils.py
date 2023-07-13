import collections.abc as cabc
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from cycler import Cycler, cycler
from loguru import logger
from matplotlib import cm, colors, rcParams
from matplotlib import pyplot as pl
from matplotlib.axes import Axes
from matplotlib.colors import is_color_like, to_hex
from pandas.api.types import is_categorical_dtype

# Colorblindness adjusted vega_10
# See https://github.com/scverse/scanpy/issues/387
vega_10 = list(map(colors.to_hex, cm.tab10.colors))
vega_10_scanpy = vega_10.copy()
vega_10_scanpy[2] = "#279e68"  # green
vega_10_scanpy[4] = "#aa40fc"  # purple
vega_10_scanpy[8] = "#b5bd61"  # kakhi
vega_20 = list(map(colors.to_hex, cm.tab20.colors))

# reorderd, some removed, some added
vega_20_scanpy = [
    # dark without grey:
    *vega_20[0:14:2],
    *vega_20[16::2],
    # light without grey:
    *vega_20[1:15:2],
    *vega_20[17::2],
    # manual additions:
    "#ad494a",
    "#8c6d31",
]
vega_20_scanpy[2] = vega_10_scanpy[2]
vega_20_scanpy[4] = vega_10_scanpy[4]
vega_20_scanpy[7] = vega_10_scanpy[8]  # kakhi shifted by missing grey
# TODO: also replace pale colors if necessary

default_20 = vega_20_scanpy

# https://graphicdesign.stackexchange.com/questions/3682/where-can-i-find-a-large-palette-set-of-contrasting-colors-for-coloring-many-d
# update 1
# orig reference http://epub.wu.ac.at/1692/1/document.pdf
zeileis_28 = [
    "#023fa5",
    "#7d87b9",
    "#bec1d4",
    "#d6bcc0",
    "#bb7784",
    "#8e063b",
    "#4a6fe3",
    "#8595e1",
    "#b5bbe3",
    "#e6afb9",
    "#e07b91",
    "#d33f6a",
    "#11c638",
    "#8dd593",
    "#c6dec7",
    "#ead3c6",
    "#f0b98d",
    "#ef9708",
    "#0fcfc0",
    "#9cded6",
    "#d5eae7",
    "#f3e1eb",
    "#f6c4e1",
    "#f79cd4",
    # these last ones were added:
    "#7f7f7f",
    "#c7c7c7",
    "#1CE6FF",
    "#336600",
]

default_28 = zeileis_28

# from http://godsnotwheregodsnot.blogspot.de/2012/09/color-distribution-methodology.html
godsnot_102 = [
    # "#000000",  # remove the black, as often, we have black colored annotation
    "#FFFF00",
    "#1CE6FF",
    "#FF34FF",
    "#FF4A46",
    "#008941",
    "#006FA6",
    "#A30059",
    "#FFDBE5",
    "#7A4900",
    "#0000A6",
    "#63FFAC",
    "#B79762",
    "#004D43",
    "#8FB0FF",
    "#997D87",
    "#5A0007",
    "#809693",
    "#6A3A4C",
    "#1B4400",
    "#4FC601",
    "#3B5DFF",
    "#4A3B53",
    "#FF2F80",
    "#61615A",
    "#BA0900",
    "#6B7900",
    "#00C2A0",
    "#FFAA92",
    "#FF90C9",
    "#B903AA",
    "#D16100",
    "#DDEFFF",
    "#000035",
    "#7B4F4B",
    "#A1C299",
    "#300018",
    "#0AA6D8",
    "#013349",
    "#00846F",
    "#372101",
    "#FFB500",
    "#C2FFED",
    "#A079BF",
    "#CC0744",
    "#C0B9B2",
    "#C2FF99",
    "#001E09",
    "#00489C",
    "#6F0062",
    "#0CBD66",
    "#EEC3FF",
    "#456D75",
    "#B77B68",
    "#7A87A1",
    "#788D66",
    "#885578",
    "#FAD09F",
    "#FF8A9A",
    "#D157A0",
    "#BEC459",
    "#456648",
    "#0086ED",
    "#886F4C",
    "#34362D",
    "#B4A8BD",
    "#00A6AA",
    "#452C2C",
    "#636375",
    "#A3C8C9",
    "#FF913F",
    "#938A81",
    "#575329",
    "#00FECF",
    "#B05B6F",
    "#8CD0FF",
    "#3B9700",
    "#04F757",
    "#C8A1A1",
    "#1E6E00",
    "#7900D7",
    "#A77500",
    "#6367A9",
    "#A05837",
    "#6B002C",
    "#772600",
    "#D790FF",
    "#9B9700",
    "#549E79",
    "#FFF69F",
    "#201625",
    "#72418F",
    "#BC23FF",
    "#99ADC0",
    "#3A2465",
    "#922329",
    "#5B4534",
    "#FDE8DC",
    "#404E55",
    "#0089A3",
    "#CB7E98",
    "#A4E804",
    "#324E72",
]

default_102 = godsnot_102

additional_colors = {
    "gold2": "#eec900",
    "firebrick3": "#cd2626",
    "khaki2": "#eee685",
    "slategray3": "#9fb6cd",
    "palegreen3": "#7ccd7c",
    "tomato2": "#ee5c42",
    "grey80": "#cccccc",
    "grey90": "#e5e5e5",
    "wheat4": "#8b7e66",
    "grey65": "#a6a6a6",
    "grey10": "#1a1a1a",
    "grey20": "#333333",
    "grey50": "#7f7f7f",
    "grey30": "#4d4d4d",
    "grey40": "#666666",
    "antiquewhite2": "#eedfcc",
    "grey77": "#c4c4c4",
    "snow4": "#8b8989",
    "chartreuse3": "#66cd00",
    "yellow4": "#8b8b00",
    "darkolivegreen2": "#bcee68",
    "olivedrab3": "#9acd32",
    "azure3": "#c1cdcd",
    "violetred": "#d02090",
    "mediumpurple3": "#8968cd",
    "purple4": "#551a8b",
    "seagreen4": "#2e8b57",
    "lightblue3": "#9ac0cd",
    "orchid3": "#b452cd",
    "indianred 3": "#cd5555",
    "grey60": "#999999",
    "mediumorchid1": "#e066ff",
    "plum3": "#cd96cd",
    "palevioletred3": "#cd6889",
}


def _validate_palette(adata: AnnData, key: str) -> None:
    """Validate palette."""
    # Checks if the list of colors in adata.uns[f'{key}_colors'] is valid
    # and updates the color list in adata.uns[f'{key}_colors'] if needed.
    # Not only valid matplotlib colors are checked but also if the color name
    # is a valid R color name, in which case it will be translated to a valid name.
    _palette: List[str] = []
    color_key = f"{key}_colors"

    for color in adata.uns[color_key]:
        if not is_color_like(color):
            # check if the color is a valid R color and translate it
            # to a valid hex color value
            if color in additional_colors:
                color = additional_colors[color]
            else:
                logger.warning(
                    f"The following color value found in adata.uns['{key}_colors'] "
                    f"is not valid: '{color!r}'. Default colors will be used instead."
                )
                _set_default_colors_for_categorical_obs(adata, adata.obs[key], key)
                _palette = []
                break
        _palette.append(color)
    # Don't modify if nothing changed
    if len(_palette) and list(_palette) != list(adata.uns[color_key]):
        adata.uns[color_key] = _palette


def _set_colors_for_categorical_obs(
    adata: AnnData,
    categories: Sequence[Union[str, int]],
    value_to_plot: str,
    palette: Union[str, Sequence[str], Cycler],
) -> None:
    """
    Set the adata.uns[value_to_plot + '_colors'] according to the given palette.

    Parameters
    ----------
    adata
        annData object
    value_to_plot
        name of a valid categorical observation
    palette
        Palette should be either a valid :func:`~matplotlib.pyplot.colormaps` string,
        a sequence of colors (in a format that can be understood by matplotlib,
        eg. RGB, RGBS, hex, or a cycler object with key='color'.

    Returns
    -------
    None
    """
    # categories = adata.obs[value_to_plot].cat.categories
    # check is palette is a valid matplotlib colormap
    if isinstance(palette, str) and palette in pl.colormaps():
        # this creates a palette from a colormap. E.g. 'Accent, Dark2, tab20'
        cmap = pl.get_cmap(palette)
        colors_list = [to_hex(x) for x in cmap(np.linspace(0, 1, len(categories)))]
    elif isinstance(palette, cabc.Mapping):
        colors_list = [to_hex(palette[k], keep_alpha=True) for k in categories]
    else:
        # check if palette is a list and convert it to a cycler, thus
        # it doesnt matter if the list is shorter than the categories length:
        if isinstance(palette, cabc.Sequence):
            if len(palette) < len(categories):
                logger.warning(
                    "Length of palette colors is smaller than the number of "
                    f"categories (palette length: {len(palette)}, "
                    f"categories length: {len(categories)}. "
                    "Some categories will have the same color."
                )
            # check that colors are valid
            _color_list = []
            for color in palette:
                if not is_color_like(color):
                    # check if the color is a valid R color and translate it
                    # to a valid hex color value
                    if color in additional_colors:
                        color = additional_colors[color]
                    else:
                        raise ValueError("The following color value of the given palette " f"is not valid: {color}")
                _color_list.append(color)

            palette = cycler(color=_color_list)
        if not isinstance(palette, Cycler):
            raise ValueError(
                "Please check that the value of 'palette' is a valid "
                "matplotlib colormap string (eg. Set2), a  list of color names "
                "or a cycler with a 'color' key."
            )
        if "color" not in palette.keys:
            raise ValueError("Please set the palette key 'color'.")

        cc = palette()
        colors_list = [to_hex(next(cc)["color"]) for x in range(len(categories))]

    adata.uns[value_to_plot + "_colors"] = colors_list


def _set_default_colors_for_categorical_obs(
    adata: AnnData, categories: Sequence[Union[str, int]], value_to_plot: str
) -> None:
    """
    Set the adata.uns[value_to_plot + '_colors'] using default color palettes.

    Parameters
    ----------
    adata
        AnnData object
    value_to_plot
        Name of a valid categorical observation
    categories
        categories of the categorical observation.

    Returns
    -------
    None
    """
    # categories = adata.obs[value_to_plot].cat.categories
    length = len(categories)

    # check if default matplotlib palette has enough colors
    if len(rcParams["axes.prop_cycle"].by_key()["color"]) >= length:
        cc = rcParams["axes.prop_cycle"]()
        palette = [next(cc)["color"] for _ in range(length)]

    else:
        if length <= 20:
            palette = default_20
        elif length <= 28:
            palette = default_28
        elif length <= len(default_102):  # 103 colors
            palette = default_102
        else:
            palette = ["grey" for _ in range(length)]
            logger.info(
                f"the obs value {value_to_plot!r} has more than 103 categories. Uniform "
                "'grey' color will be used for all categories."
            )

    _set_colors_for_categorical_obs(adata, categories, value_to_plot, palette[:length])


def add_colors_for_categorical_sample_annotation(
    adata: AnnData, key: str, vec: pd.Series, palette: Optional[List[str]] = None, force_update_colors: bool = False
) -> None:
    """Add colors for categorical annotation."""
    color_key = f"{key}_colors"
    if not is_categorical_dtype(adata.obs[key]) and is_categorical_dtype(vec):
        categories = vec.cat.categories
    elif is_categorical_dtype(adata.obs[key]):
        categories = adata.obs[key].cat.categories
    colors_needed = len(categories)
    if palette and force_update_colors:
        _set_colors_for_categorical_obs(adata, categories, key, palette)
    elif color_key in adata.uns and len(adata.uns[color_key]) <= colors_needed:
        _validate_palette(adata, key)
    else:
        _set_default_colors_for_categorical_obs(adata, categories, key)


def _add_categorical_legend(
    ax: Axes,
    color_source_vector: pd.Series,
    palette: Dict[str, str],
    legend_loc: str = "right margin",
    legend_fontweight: str = "bold",
    legend_fontsize: Optional[float] = None,
    legend_fontoutline: Optional[float] = None,
    multi_panel: bool = False,
    na_color: str = "lightgray",
    na_in_legend: bool = True,
    scatter_array: Optional[Any] = None,  # added defaults compared to scanpy
) -> None:
    """Add a legend to the passed Axes."""
    if na_in_legend and pd.isnull(color_source_vector).any():
        if "NA" in color_source_vector:
            raise NotImplementedError("No fallback for null labels has been defined if NA already in categories.")
        color_source_vector = color_source_vector.add_categories("NA").fillna("NA")
        palette = palette.copy()
        palette["NA"] = na_color
    cats = color_source_vector.cat.categories  # changed compared to original function from scanpy

    if multi_panel is True:
        # Shrink current axis by 10% to fit legend and match
        # size of plots that are not categorical
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.91, box.height])

    if legend_loc == "right margin":
        for label in cats:
            ax.scatter([], [], c=palette[label], label=label)
        ax.legend(
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            ncol=(1 if len(cats) <= 14 else 2 if len(cats) <= 30 else 3),
            fontsize=legend_fontsize,
        )
    elif legend_loc == "on data":
        # identify centroids to put labels

        all_pos = (
            pd.DataFrame(scatter_array, columns=["x", "y"])
            .groupby(color_source_vector, observed=True)
            .median()
            # Have to sort_index since if observed=True and categorical is unordered
            # the order of values in .index is undefined. Related issue:
            # https://github.com/pandas-dev/pandas/issues/25167
            .sort_index()
        )

        for label, x_pos, y_pos in all_pos.itertuples():
            ax.text(
                x_pos,
                y_pos,
                label,
                weight=legend_fontweight,
                verticalalignment="center",
                horizontalalignment="center",
                fontsize=legend_fontsize,
                path_effects=legend_fontoutline,
            )
