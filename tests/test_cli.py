from pathlib import Path

import pytest
from click.testing import CliRunner
from napari.viewer import Viewer
from spatialdata.datasets import blobs

from napari_spatialdata.__main__ import cli


def test_view_exists():
    runner = CliRunner()
    result = runner.invoke(cli, ["view"])
    assert result.exit_code == 2  # Error because of missing argument
    Viewer.close_all()


@pytest.mark.usefixtures("mock_app_model")
def test_view_path_not_exists():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["view", "non_existent_path.zarr"])
        assert isinstance(result.exception, AssertionError) and result.exception.args[0].endswith(
            "non_existent_path.zarr does not exist"
        )
        assert result.exit_code == 1

        Viewer.close_all()


# added due to this https://github.com/napari/napari/issues/8214#issuecomment-3188565917
@pytest.mark.usefixtures("mock_app_model")
def test_view_path_is_dir():
    runner = CliRunner()
    with runner.isolated_filesystem():
        f = Path("data.zarr")
        blobs().write(f)
        result = runner.invoke(cli, ["view", "data.zarr", "--headless"])
        assert result.exit_code == 0  # Command executed successfully

        # Close all existing viewer instances to avoid leaking of viewers between tests
        Viewer.close_all()
