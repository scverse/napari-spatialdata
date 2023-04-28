from napari_spatialdata import Interactive
from spatialdata import SpatialData
#from spatialdata.datasets import blobs

if __name__ == "__main__":
    sdata = SpatialData.read("../data/xenium/data.zarr")
    #sdata = blobs()
    i = Interactive(sdata)
    i.run()
