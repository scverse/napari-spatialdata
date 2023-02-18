import napari
import squidpy as sq

adata = sq.datasets.visium_hne_adata()
img1 = adata.uns["spatial"]["V1_Adult_Mouse_Brain"]["images"]["hires"].copy()

viewer = napari.Viewer()
viewer.add_image(
    img1,
    rgb=True,
    name="image1",
    colormap="inferno",
    metadata={"adata": adata},
    scale=(1, 1),
)

viewer.scale_bar.visible = True
viewer.scale_bar.unit = "um"

if __name__ == "__main__":
    napari.run()
