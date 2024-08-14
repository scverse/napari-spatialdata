from napari_spatialdata._model import DataModel


class DataModelSuite:
    def setup(self) -> None:
        self.model = DataModel()

    def mem_model(self) -> DataModel:
        return self.model

    def time_model_get_items(self) -> None:
        self.model.get_items("columns_df")
