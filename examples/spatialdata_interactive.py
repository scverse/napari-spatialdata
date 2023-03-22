from spatialdata import SpatialData

from napari_spatialdata._interactive import Interactive

if __name__ == "__main__":
    sdata = SpatialData.read("../data/cosmx/data.zarr")
    sdata.table.uns["spatialdata_attrs"]["region"] = 0
    i = Interactive(sdata)
    i.run()
