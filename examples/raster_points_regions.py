import napari
import anndata as ad

adata = ad.read("./adata_nanostring_points.h5ad")

img1 = adata.uns["spatial"]["1"]["images"]["hires"]
img2 = adata.uns["spatial"]["2"]["images"]["hires"]

label1 = adata.uns["spatial"]["1"]["images"]["segmentation"]
label2 = adata.uns["spatial"]["2"]["images"]["segmentation"]

adata1 = adata[adata.obs.fov == "1"].copy()
adata2 = adata[adata.obs.fov == "2"].copy()

points1 = adata.uns["spatial"]["1"]["points"]
points2 = adata.uns["spatial"]["2"]["points"]

viewer = napari.Viewer()
viewer.add_image(
    img1,
    rgb=True,
    name="image1",
)
viewer.add_labels(
    label1,
    name="label1",
    metadata={
        "adata": adata1,
        "library_id": "1",
        "labels_key": "cell_ID",
        "points": points1,
        "point_diameter": 10,
    },  # adata in labels layer will color segments
)
viewer.add_image(
    img2,
    rgb=True,
    name="image2",
)
viewer.add_labels(
    label2,
    name="label2",
    metadata={
        "adata": adata2,
        "library_id": "2",
        "labels_key": "cell_ID",
        "points": points2,
        "point_diameter": 10,
    },  # adata in labels layer will color segments
)

if __name__ == "__main__":
    napari.run()
