import sys

from scanpy import settings as s
from anndata import AnnData
from scipy.sparse import issparse
import numpy as np
import pytest

from tests.conftest import DPI, PlotTester, PlotTesterMeta
from napari_spatialdata._container import Container
from napari_spatialdata.interactive import Interactive


@pytest.mark.qt()
class TestNapari(PlotTester, metaclass=PlotTesterMeta):
    def test_add_same_layer(self, qtbot, adata: AnnData, napari_cont: Container, capsys):
        from napari.layers import Points

        s.logfile = sys.stderr
        s.verbosity = 4

        viewer = Interactive(napari_cont, adata)
        cnt = viewer._controller

        data = np.random.normal(size=adata.n_obs)
        cnt.add_points(data, layer_name="layer1")
        cnt.add_points(np.random.normal(size=adata.n_obs), layer_name="layer1")

        err = capsys.readouterr().err

        assert "Layer `layer1` is already loaded" in err
        assert len(viewer._controller.view.layers) == 2
        assert viewer._controller.view.layernames == {"V1_Adult_Mouse_Brain", "layer1"}
        assert isinstance(viewer._controller.view.layers["layer1"], Points)
        np.testing.assert_array_equal(viewer._controller.view.layers["layer1"].metadata["data"], data)

    def test_add_not_categorical_series(self, qtbot, adata: AnnData, napari_cont: Container):
        viewer = Interactive(napari_cont, adata)
        cnt = viewer._controller

        with pytest.raises(TypeError, match=r"Expected a `categorical` type,.*"):
            cnt.add_points(adata.obs["in_tissue"].astype(int), layer_name="layer1")

    def test_plot_simple_canvas(self, qtbot, adata: AnnData, napari_cont: Container):
        viewer = Interactive(napari_cont, adata)

        viewer.screenshot(dpi=DPI)

    def test_plot_symbol(self, qtbot, adata: AnnData, napari_cont: Container):
        viewer = Interactive(napari_cont, adata, symbol="square")
        cnt = viewer._controller

        cnt.add_points(adata.obs_vector(adata.var_names[42]), layer_name="foo")
        viewer.screenshot(dpi=DPI)

    def test_plot_gene_X(self, qtbot, adata: AnnData, napari_cont: Container):
        viewer = Interactive(napari_cont, adata)
        cnt = viewer._controller

        cnt.add_points(adata.obs_vector(adata.var_names[42]), layer_name="foo")
        viewer.screenshot(dpi=DPI)

    def test_plot_obs_continuous(self, qtbot, adata: AnnData, napari_cont: Container):
        viewer = Interactive(napari_cont, adata)
        cnt = viewer._controller

        cnt.add_points(np.random.RandomState(42).normal(size=adata.n_obs), layer_name="quux")
        viewer.screenshot(dpi=DPI)

    def test_plot_obs_categorical(self, qtbot, adata: AnnData, napari_cont: Container):
        viewer = Interactive(napari_cont, adata)
        cnt = viewer._controller

        cnt.add_points(adata.obs["leiden"], key="leiden", layer_name="quas")
        viewer.screenshot(dpi=DPI)

    def test_plot_cont_cmap(self, qtbot, adata: AnnData, napari_cont: Container):
        viewer = Interactive(napari_cont, adata, cmap="inferno")
        cnt = viewer._controller

        cnt.add_points(adata.obs_vector(adata.var_names[42]), layer_name="wex")
        viewer.screenshot(dpi=DPI)

    def test_plot_cat_cmap(self, qtbot, adata: AnnData, napari_cont: Container):
        viewer = Interactive(napari_cont, adata, palette="Set3")
        cnt = viewer._controller

        cnt.add_points(adata.obs["leiden"].astype("category"), key="in_tissue", layer_name="exort")
        viewer.screenshot(dpi=DPI)

    def test_plot_blending(self, qtbot, adata: AnnData, napari_cont: Container):
        viewer = Interactive(napari_cont, adata, blending="additive")
        cnt = viewer._controller

        for gene in adata.var_names[42:46]:
            data = adata.obs_vector(gene)
            if issparse(data):  # ImageModel handles sparsity, here we have to do it ourselves
                data = data.X
            cnt.add_points(data, layer_name=gene)

        viewer.screenshot(dpi=DPI)

    def test_plot_scalefactor(self, qtbot, adata: AnnData, napari_cont: Container):
        scale = 2
        napari_cont.data.attrs["scale"] = scale

        viewer = Interactive(napari_cont, adata)
        cnt = viewer._controller
        model = cnt._model

        data = np.random.normal(size=adata.n_obs)
        cnt.add_points(data, layer_name="layer1")

        # ignore z-dim
        np.testing.assert_allclose(adata.obsm["spatial"][:, ::-1] * scale, model.coordinates[:, 1:])

        viewer.screenshot(dpi=DPI)
