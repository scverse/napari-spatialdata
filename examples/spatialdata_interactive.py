from napari_spatialdata._interactive import Interactive
from spatialdata import SpatialData

if __name__ == "__main__":
    sdata = SpatialData.read("../data/cosmx/data.zarr")
    sdata.table.uns["spatialdata_attrs"]["region"] = 0
    i = Interactive(sdata)
    i.run()
