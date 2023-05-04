# In this script, we are reading the MibiTOF dataset into a SpatialData object and visualising
# it with the Interactive class from napari_spatialdata.

# The dataset can be downloaded from https://spatialdata.scverse.org/en/latest/tutorials/notebooks/datasets/README.html

from napari_spatialdata import Interactive
from spatialdata import SpatialData

if __name__ == "__main__":
    sdata = SpatialData.read("../data/mibitof/data.zarr")  # Change this path!
    i = Interactive(sdata)
    i.run()
