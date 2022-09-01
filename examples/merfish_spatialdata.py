##
import spatialdata as sd
from pathlib import Path

# use symlinks to make the spatialdata-sandbox datasets repository available
data_dir = Path("spatialdata-sandbox/merfish/data.zarr")
assert data_dir.exists()
sdata = sd.SpatialData.read(data_dir)
print(sdata)

from napari_spatialdata import Interactive
interactive = Interactive(sdata=sdata)