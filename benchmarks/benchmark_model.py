import os

from napari_spatialdata._model import DataModel


class DataModelSuite:
    params = [1] if "PR" in os.environ else [1, 10]

    def setup(self, _n: int) -> None:
        self.model = DataModel()

    def mem_model(self, _n: int) -> DataModel:
        return self.model

    def time_model_get_items(self, _n: int) -> None:
        self.model.get_items("columns_df")
