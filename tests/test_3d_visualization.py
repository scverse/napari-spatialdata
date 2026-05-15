"""Tests for 3D points, 2.5D shapes and 2.5D circles visualization.

For debugging tips on how to visually inspect tests, see docs/contributing.md.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from napari.layers import Points, Shapes
from napari.utils.events import EventedList
from spatialdata import SpatialData
from spatialdata.models import get_axes_names

from napari_spatialdata._sdata_widgets import SdataWidget
from napari_spatialdata.constants import config


class Test3DPointsVisualization:
    """Test 3D points visualization in napari."""

    @pytest.mark.parametrize(
        ("project_to_2d", "expected_ndim"),
        [(True, 2), (False, 3)],
        ids=["projected_to_2d", "full_3d"],
    )
    def test_3d_points_visualization(
        self,
        make_napari_viewer: Any,
        sdata_3d_points: SpatialData,
        project_to_2d: bool,
        expected_ndim: int,
    ):
        """Points dimensionality follows the ``PROJECT_3D_POINTS_TO_2D`` config flag."""
        original_value = config.PROJECT_3D_POINTS_TO_2D
        try:
            config.PROJECT_3D_POINTS_TO_2D = project_to_2d

            viewer = make_napari_viewer()
            widget = SdataWidget(viewer, EventedList([sdata_3d_points]))

            widget.coordinate_system_widget._select_coord_sys("global")
            widget.elements_widget._onItemChange("global")
            widget._onClick("points_3d")
            viewer.dims.ndisplay = 3

            assert len(viewer.layers) == 1
            assert isinstance(viewer.layers[0], Points)
            assert viewer.layers[0].data.shape[1] == expected_ndim
        finally:
            config.PROJECT_3D_POINTS_TO_2D = original_value


class Test2_5DShapesVisualization:
    """Test 2.5D shapes visualization in napari."""

    @pytest.mark.parametrize(
        ("project_to_2d", "expected_ndim"),
        [(True, 2), (False, 3)],
        ids=["projected_to_2d", "full_3d"],
    )
    def test_2_5d_shapes_visualization(
        self,
        make_napari_viewer: Any,
        sdata_2_5d_shapes: SpatialData,
        project_to_2d: bool,
        expected_ndim: int,
    ):
        """Shape vertex dimensionality follows the ``PROJECT_2_5D_SHAPES_TO_2D`` config flag."""
        original_value = config.PROJECT_2_5D_SHAPES_TO_2D
        try:
            config.PROJECT_2_5D_SHAPES_TO_2D = project_to_2d

            viewer = make_napari_viewer()
            widget = SdataWidget(viewer, EventedList([sdata_2_5d_shapes]))

            widget.coordinate_system_widget._select_coord_sys("global")
            widget.elements_widget._onItemChange("global")
            widget._onClick("shapes_2.5d")

            assert len(viewer.layers) == 1
            assert isinstance(viewer.layers[0], Shapes)
            for shape_data in viewer.layers[0].data:
                assert shape_data.shape[1] == expected_ndim
        finally:
            config.PROJECT_2_5D_SHAPES_TO_2D = original_value


class Test2_5DCirclesVisualization:
    """Test 2.5D circles visualization in napari."""

    @pytest.mark.parametrize(
        ("project_to_2d", "expected_ndim"),
        [(True, 2), (False, 3)],
        ids=["projected_to_2d", "full_3d"],
    )
    def test_2_5d_circles_visualization(
        self,
        make_napari_viewer: Any,
        sdata_2_5d_circles: SpatialData,
        project_to_2d: bool,
        expected_ndim: int,
    ):
        """Circles dimensionality follows the ``PROJECT_2_5D_SHAPES_TO_2D`` config flag."""
        original_value = config.PROJECT_2_5D_SHAPES_TO_2D
        try:
            config.PROJECT_2_5D_SHAPES_TO_2D = project_to_2d

            viewer = make_napari_viewer()
            widget = SdataWidget(viewer, EventedList([sdata_2_5d_circles]))

            widget.coordinate_system_widget._select_coord_sys("global")
            widget.elements_widget._onItemChange("global")
            widget._onClick("circles_2.5d")

            assert len(viewer.layers) == 1
            assert viewer.layers[0].data.shape[1] == expected_ndim
        finally:
            config.PROJECT_2_5D_SHAPES_TO_2D = original_value


class TestAffineTransformLayers:
    """Test that ``_affine_transform_layers`` propagates ``include_z`` correctly."""

    @pytest.mark.parametrize(
        ("project_to_2d", "expected_data_ndim", "expected_affine_shape"),
        [(False, 3, (4, 4)), (True, 2, (3, 3))],
        ids=["full_3d", "projected_to_2d"],
    )
    def test_affine_transform_preserves_dimensionality(
        self,
        make_napari_viewer: Any,
        sdata_3d_points_two_cs: SpatialData,
        project_to_2d: bool,
        expected_data_ndim: int,
        expected_affine_shape: tuple[int, int],
    ):
        """Switching coordinate system preserves the affine matrix dimensionality."""
        original_value = config.PROJECT_3D_POINTS_TO_2D
        try:
            config.PROJECT_3D_POINTS_TO_2D = project_to_2d

            viewer = make_napari_viewer()
            widget = SdataWidget(viewer, EventedList([sdata_3d_points_two_cs]))

            widget.coordinate_system_widget._select_coord_sys("global")
            widget.elements_widget._onItemChange("global")
            widget._onClick("points_3d")

            assert len(viewer.layers) == 1
            layer = viewer.layers[0]
            assert isinstance(layer, Points)
            assert layer.data.shape[1] == expected_data_ndim

            # Identity in "global" -> affine should be the identity of the expected shape
            np.testing.assert_array_almost_equal(layer.affine.affine_matrix, np.eye(expected_affine_shape[0]))

            widget.coordinate_system_widget._select_coord_sys("scaled")
            widget.viewer_model._affine_transform_layers("scaled")

            # After switching the affine must keep its dimensionality
            assert layer.affine.affine_matrix.shape == expected_affine_shape
            if not project_to_2d:
                assert not np.allclose(layer.affine.affine_matrix, np.eye(expected_affine_shape[0]))
        finally:
            config.PROJECT_3D_POINTS_TO_2D = original_value


class TestSavePointsPreservesZ:
    """Test that saving points correctly handles the z coordinate."""

    @pytest.mark.parametrize(
        ("project_to_2d", "expected_data_ndim", "z_in_axes"),
        [(False, 3, True), (True, 2, False)],
        ids=["preserve_z", "drop_z"],
    )
    def test_save_points_z_handling(
        self,
        tmp_path: Path,
        make_napari_viewer: Any,
        sdata_3d_points: SpatialData,
        project_to_2d: bool,
        expected_data_ndim: int,
        z_in_axes: bool,
    ):
        """Saving a 3D points layer must retain or drop the z column based on the config flag."""
        original_value = config.PROJECT_3D_POINTS_TO_2D
        try:
            config.PROJECT_3D_POINTS_TO_2D = project_to_2d

            tmpdir = tmp_path / "sdata.zarr"
            sdata_3d_points.write(tmpdir)

            viewer = make_napari_viewer()
            widget = SdataWidget(viewer, EventedList([sdata_3d_points]))

            widget.coordinate_system_widget._select_coord_sys("global")
            widget.elements_widget._onItemChange("global")
            widget._onClick("points_3d")

            layer = viewer.layers[0]
            assert isinstance(layer, Points)
            assert layer.data.shape[1] == expected_data_ndim

            parsed, _ = widget.viewer_model._save_points_to_sdata(layer, "points_3d", overwrite=True)

            saved_axes = get_axes_names(parsed)
            assert ("z" in saved_axes) is z_in_axes

            if z_in_axes:
                original_z = sdata_3d_points.points["points_3d"].compute()["z"].values
                saved_z = parsed.compute()["z"].values
                np.testing.assert_array_almost_equal(saved_z, original_z)
        finally:
            config.PROJECT_3D_POINTS_TO_2D = original_value


class TestSaveShapesPreservesZ:
    """Test that saving shapes correctly handles the z coordinate."""

    @pytest.mark.parametrize(
        ("project_to_2d", "expected_vertex_ndim", "z_in_axes"),
        [(False, 3, True), (True, 2, False)],
        ids=["preserve_z", "drop_z"],
    )
    def test_save_shapes_z_handling(
        self,
        tmp_path: Path,
        make_napari_viewer: Any,
        sdata_2_5d_shapes: SpatialData,
        project_to_2d: bool,
        expected_vertex_ndim: int,
        z_in_axes: bool,
    ):
        """Saving a 2.5D shapes layer must retain or drop the z column based on the config flag."""
        original_value = config.PROJECT_2_5D_SHAPES_TO_2D
        try:
            config.PROJECT_2_5D_SHAPES_TO_2D = project_to_2d

            tmpdir = tmp_path / "sdata.zarr"
            sdata_2_5d_shapes.write(tmpdir)

            viewer = make_napari_viewer()
            widget = SdataWidget(viewer, EventedList([sdata_2_5d_shapes]))

            widget.coordinate_system_widget._select_coord_sys("global")
            widget.elements_widget._onItemChange("global")
            widget._onClick("shapes_2.5d")

            layer = viewer.layers[0]
            assert isinstance(layer, Shapes)
            for shape_data in layer.data:
                assert shape_data.shape[1] == expected_vertex_ndim

            parsed, _ = widget.viewer_model._save_shapes_to_sdata(layer, "shapes_2.5d", overwrite=True)

            saved_axes = get_axes_names(parsed)
            assert ("z" in saved_axes) is z_in_axes

            if z_in_axes:
                saved_z = parsed["z"].values
                original_unique_z = np.unique(sdata_2_5d_shapes.shapes["shapes_2.5d"]["z"].values)
                np.testing.assert_array_almost_equal(np.unique(saved_z), original_unique_z)
        finally:
            config.PROJECT_2_5D_SHAPES_TO_2D = original_value


class TestUIToggle:
    """Test the 3D settings checkboxes in SdataWidget."""

    def test_toggle_affects_loaded_points(
        self,
        make_napari_viewer: Any,
        sdata_3d_points: SpatialData,
    ):
        """Toggling the checkbox affects the dimensionality of newly loaded layers.

        This implicitly also tests that the checkbox state and the underlying
        ``config.PROJECT_3D_POINTS_TO_2D`` flag stay in sync.
        """
        original_value = config.PROJECT_3D_POINTS_TO_2D
        try:
            config.PROJECT_3D_POINTS_TO_2D = True

            viewer = make_napari_viewer()
            widget = SdataWidget(viewer, EventedList([sdata_3d_points]))

            widget.coordinate_system_widget._select_coord_sys("global")
            widget.elements_widget._onItemChange("global")

            widget._onClick("points_3d")
            assert viewer.layers[0].data.shape[1] == 2

            viewer.layers.clear()

            widget.discard_z_points.setChecked(False)
            widget._onClick("points_3d")
            assert viewer.layers[0].data.shape[1] == 3
        finally:
            config.PROJECT_3D_POINTS_TO_2D = original_value


class TestMixed2D3DVisualization:
    """Test mixed 2D and 3D visualization scenarios."""

    @pytest.mark.parametrize(
        "points_dim,shapes_dim",
        [
            (3, 2),  # Points 3D, Shapes 2D
            (2, 3),  # Points 2D, Shapes 3D
        ],
    )
    def test_mixed_dimension_visualization(
        self,
        make_napari_viewer: Any,
        sdata_3d_points: SpatialData,
        sdata_2_5d_shapes: SpatialData,
        points_dim: int,
        shapes_dim: int,
    ):
        """Test that points and shapes can be visualized with different dimension settings."""
        original_points_config = config.PROJECT_3D_POINTS_TO_2D
        original_shapes_config = config.PROJECT_2_5D_SHAPES_TO_2D

        try:
            config.PROJECT_3D_POINTS_TO_2D = points_dim == 2
            config.PROJECT_2_5D_SHAPES_TO_2D = shapes_dim == 2

            combined_sdata = SpatialData(
                points={"points_3d": sdata_3d_points["points_3d"]},
                shapes={"shapes_2.5d": sdata_2_5d_shapes["shapes_2.5d"]},
            )

            viewer = make_napari_viewer()
            widget = SdataWidget(viewer, EventedList([combined_sdata]))

            widget.coordinate_system_widget._select_coord_sys("global")
            widget.elements_widget._onItemChange("global")

            widget._onClick("points_3d")
            assert viewer.layers[0].data.shape[1] == points_dim

            widget._onClick("shapes_2.5d")
            for shape_data in viewer.layers[1].data:
                assert shape_data.shape[1] == shapes_dim

        finally:
            config.PROJECT_3D_POINTS_TO_2D = original_points_config
            config.PROJECT_2_5D_SHAPES_TO_2D = original_shapes_config
