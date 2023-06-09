from pathlib import Path

from click.testing import CliRunner
from napari.viewer import Viewer
from napari_spatialdata.__main__ import cli
from spatialdata.datasets import blobs


def test_view_exists():
    runner = CliRunner()
    result = runner.invoke(cli, ["view"])
    assert result.exit_code == 2  # Error because of missing argument


def test_view_path_not_exists():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["view", "non_existent_path.zarr"])
        assert isinstance(result.exception, AssertionError) and result.exception.args[0].endswith(
            "non_existent_path.zarr does not exist"
        )
        assert result.exit_code == 1


def test_view_path_is_dir():
    runner = CliRunner()
    with runner.isolated_filesystem():
        f = Path("data.zarr")
        blobs().write(f)
        result = runner.invoke(cli, ["view", "data.zarr", "--headless"])
        assert result.exit_code == 0  # Command executed successfully

        # Close all existing viewer instances to avoid leaking of viewers between tests
        Viewer.close_all()
