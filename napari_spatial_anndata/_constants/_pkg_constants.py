"""Internal constants not exposed to the user."""
from typing import Any, Callable, Optional

from anndata import AnnData

_SEP = "_"


class cprop:
    def __init__(self, f: Callable[..., str]):
        self.f = f

    def __get__(self, obj: Any, owner: Any) -> str:
        return self.f(owner)


class Key:
    class img:
        @cprop
        def coords(cls) -> str:
            return "coords"

        @cprop
        def padding(cls) -> str:
            return "padding"

        @cprop
        def mask_circle(cls) -> str:
            return "mask_circle"

        @cprop
        def scale(self) -> str:
            return "scale"

        @cprop
        def obs(cls) -> str:
            return "cell"

    class obs:
        pass

    class obsm:
        @cprop
        def spatial(cls) -> str:
            return "spatial"

    class uns:
        @cprop
        def spatial(cls) -> str:
            return Key.obsm.spatial

        @cprop
        def image_key(cls) -> str:
            return "images"

        @cprop
        def image_res_key(cls) -> str:
            return "hires"

        @cprop
        def image_seg_key(cls) -> str:
            return "segmentation"

        @cprop
        def scalefactor_key(cls) -> str:
            return "scalefactors"

        @cprop
        def size_key(cls) -> str:
            return "spot_diameter_fullres"

        @classmethod
        def spatial_neighs(cls, value: Optional[str] = None) -> str:
            return f"{Key.obsm.spatial}_neighbors" if value is None else f"{value}_neighbors"

        @classmethod
        def colors(cls, cluster: str) -> str:
            return f"{cluster}_colors"

        @classmethod
        def spot_diameter(cls, adata: AnnData, spatial_key: str, library_id: Optional[str] = None) -> float:
            try:
                return float(adata.uns[spatial_key][library_id]["scalefactors"]["spot_diameter_fullres"])
            except KeyError:
                raise KeyError(
                    f"Unable to get the spot diameter from "
                    f"`adata.uns[{spatial_key!r}][{library_id!r}]['scalefactors']['spot_diameter_fullres']]`"
                ) from None

    class obsp:
        @classmethod
        def spatial_dist(cls, value: Optional[str] = None) -> str:
            return f"{Key.obsm.spatial}_distances" if value is None else f"{value}_distances"

        @classmethod
        def spatial_conn(cls, value: Optional[str] = None) -> str:
            return f"{Key.obsm.spatial}_connectivities" if value is None else f"{value}_connectivities"
