import napari
import squidpy as sq

adata = sq.datasets.visium_hne_adata()
img1 = adata.uns["spatial"]["V1_Adult_Mouse_Brain"]["images"]["hires"].copy()

viewer = napari.Viewer()
viewer.add_image(
    img1,
    rgb=True,
    name="image1",
    metadata={"adata": adata, "library_id": "V1_Adult_Mouse_Brain"},
)

if __name__ == "__main__":
    napari.run()
