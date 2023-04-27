from napari_spatialdata import Interactive
from spatialdata import SpatialData

if __name__ == "__main__":
    sdata = SpatialData.read("../data/cosmx/data.zarr")
    i = Interactive(sdata)
    i.run()
