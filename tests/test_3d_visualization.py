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

    def test_3d_points_projected_to_2d(self, make_napari_viewer: Any, sdata_3d_points: SpatialData):
        """Test that 3D points are projected to 2D when config flag is True."""
        original_value = config.PROJECT_3D_POINTS_TO_2D
        try:
            config.PROJECT_3D_POINTS_TO_2D = True

            viewer = make_napari_viewer()
            widget = SdataWidget(viewer, EventedList([sdata_3d_points]))

            widget.coordinate_system_widget._select_coord_sys("global")
            widget.elements_widget._onItemChange("global")
            widget._onClick("points_3d")
            viewer.dims.ndisplay = 3

            assert len(viewer.layers) == 1
            assert isinstance(viewer.layers[0], Points)
            # 2D projection: points should have 2 coordinates
            assert viewer.layers[0].data.shape[1] == 2
        finally:
            config.PROJECT_3D_POINTS_TO_2D = original_value

    def test_3d_points_full_3d(self, make_napari_viewer: Any, sdata_3d_points: SpatialData):
        """Test that 3D points are visualized in 3D when config flag is False."""
        original_value = config.PROJECT_3D_POINTS_TO_2D
        try:
            config.PROJECT_3D_POINTS_TO_2D = False

            viewer = make_napari_viewer()
            widget = SdataWidget(viewer, EventedList([sdata_3d_points]))

            widget.coordinate_system_widget._select_coord_sys("global")
            widget.elements_widget._onItemChange("global")
            widget._onClick("points_3d")
            viewer.dims.ndisplay = 3

            assert len(viewer.layers) == 1
            assert isinstance(viewer.layers[0], Points)
            # Full 3D: points should have 3 coordinates (z, y, x)
            assert viewer.layers[0].data.shape[1] == 3
        finally:
            config.PROJECT_3D_POINTS_TO_2D = original_value


class Test2_5DShapesVisualization:
    """Test 2.5D shapes visualization in napari."""

    def test_2_5d_shapes_projected_to_2d(self, make_napari_viewer: Any, sdata_2_5d_shapes: SpatialData):
        """Test that 2.5D shapes are projected to 2D when config flag is True."""
        original_value = config.PROJECT_2_5D_SHAPES_TO_2D
        try:
            config.PROJECT_2_5D_SHAPES_TO_2D = True

            viewer = make_napari_viewer()
            widget = SdataWidget(viewer, EventedList([sdata_2_5d_shapes]))

            widget.coordinate_system_widget._select_coord_sys("global")
            widget.elements_widget._onItemChange("global")

            # Add 2.5D shapes
            widget._onClick("shapes_2.5d")

            assert len(viewer.layers) == 1
            assert isinstance(viewer.layers[0], Shapes)
            # 2D projection: shape coordinates should have 2 values per vertex (y, x)
            for shape_data in viewer.layers[0].data:
                assert shape_data.shape[1] == 2

        finally:
            config.PROJECT_2_5D_SHAPES_TO_2D = original_value

    def test_2_5d_shapes_full_3d(self, make_napari_viewer: Any, sdata_2_5d_shapes: SpatialData):
        """Test that 2.5D shapes are visualized in 3D when config flag is False."""
        original_value = config.PROJECT_2_5D_SHAPES_TO_2D
        try:
            config.PROJECT_2_5D_SHAPES_TO_2D = False

            viewer = make_napari_viewer()
            widget = SdataWidget(viewer, EventedList([sdata_2_5d_shapes]))

            widget.coordinate_system_widget._select_coord_sys("global")
            widget.elements_widget._onItemChange("global")

            # Add 2.5D shapes
            widget._onClick("shapes_2.5d")

            assert len(viewer.layers) == 1
            assert isinstance(viewer.layers[0], Shapes)
            # Full 3D: shape coordinates should have 3 values per vertex (z, y, x)
            for shape_data in viewer.layers[0].data:
                assert shape_data.shape[1] == 3
        finally:
            config.PROJECT_2_5D_SHAPES_TO_2D = original_value


class Test2_5DCirclesVisualization:
    """Test 2.5D circles visualization in napari."""

    def test_2_5d_circles_projected_to_2d(self, make_napari_viewer: Any, sdata_2_5d_circles: SpatialData):
        """Test that 2.5D circles are projected to 2D when config flag is True."""
        original_value = config.PROJECT_2_5D_SHAPES_TO_2D
        try:
            config.PROJECT_2_5D_SHAPES_TO_2D = True

            viewer = make_napari_viewer()
            widget = SdataWidget(viewer, EventedList([sdata_2_5d_circles]))

            widget.coordinate_system_widget._select_coord_sys("global")
            widget.elements_widget._onItemChange("global")
            widget._onClick("circles_2.5d")

            assert len(viewer.layers) == 1
            # 2D projection: coordinates should have 2 values (y, x)
            assert viewer.layers[0].data.shape[1] == 2
        finally:
            config.PROJECT_2_5D_SHAPES_TO_2D = original_value

    def test_2_5d_circles_full_3d(self, make_napari_viewer: Any, sdata_2_5d_circles: SpatialData):
        """Test that 2.5D circles are visualized in 3D when config flag is False."""
        original_value = config.PROJECT_2_5D_SHAPES_TO_2D
        try:
            config.PROJECT_2_5D_SHAPES_TO_2D = False

            viewer = make_napari_viewer()
            widget = SdataWidget(viewer, EventedList([sdata_2_5d_circles]))

            widget.coordinate_system_widget._select_coord_sys("global")
            widget.elements_widget._onItemChange("global")
            widget._onClick("circles_2.5d")

            assert len(viewer.layers) == 1
            # Full 3D: coordinates should have 3 values (z, y, x)
            assert viewer.layers[0].data.shape[1] == 3
        finally:
            config.PROJECT_2_5D_SHAPES_TO_2D = original_value


class TestAffineTransformLayers:
    """Test that _affine_transform_layers propagates include_z correctly."""

    def test_affine_transform_preserves_3d_for_points(
        self,
        make_napari_viewer: Any,
        sdata_3d_points_two_cs: SpatialData,
    ):
        """Switching coordinate system must produce a 4x4 affine for 3D points."""
        original_value = config.PROJECT_3D_POINTS_TO_2D
        try:
            config.PROJECT_3D_POINTS_TO_2D = False

            viewer = make_napari_viewer()
            widget = SdataWidget(viewer, EventedList([sdata_3d_points_two_cs]))

            widget.coordinate_system_widget._select_coord_sys("global")
            widget.elements_widget._onItemChange("global")
            widget._onClick("points_3d")

            assert len(viewer.layers) == 1
            layer = viewer.layers[0]
            assert isinstance(layer, Points)
            assert layer.data.shape[1] == 3

            # Identity in "global" -> affine should be 4x4 identity
            np.testing.assert_array_almost_equal(layer.affine.affine_matrix, np.eye(4))

            # Switch to the "scaled" coordinate system
            widget.coordinate_system_widget._select_coord_sys("scaled")
            widget.viewer_model._affine_transform_layers("scaled")

            # After switching, the affine must still be 4x4 (not 3x3)
            assert layer.affine.affine_matrix.shape == (4, 4)
            assert not np.allclose(layer.affine.affine_matrix, np.eye(4))
        finally:
            config.PROJECT_3D_POINTS_TO_2D = original_value

    def test_affine_transform_projects_to_2d_when_configured(
        self,
        make_napari_viewer: Any,
        sdata_3d_points_two_cs: SpatialData,
    ):
        """When projection is enabled the affine must be 3x3 (2D)."""
        original_value = config.PROJECT_3D_POINTS_TO_2D
        try:
            config.PROJECT_3D_POINTS_TO_2D = True

            viewer = make_napari_viewer()
            widget = SdataWidget(viewer, EventedList([sdata_3d_points_two_cs]))

            widget.coordinate_system_widget._select_coord_sys("global")
            widget.elements_widget._onItemChange("global")
            widget._onClick("points_3d")

            assert len(viewer.layers) == 1
            layer = viewer.layers[0]
            assert isinstance(layer, Points)
            assert layer.data.shape[1] == 2

            widget.coordinate_system_widget._select_coord_sys("scaled")
            widget.viewer_model._affine_transform_layers("scaled")

            # Projected to 2D -> affine stays 3x3
            assert layer.affine.affine_matrix.shape == (3, 3)
        finally:
            config.PROJECT_3D_POINTS_TO_2D = original_value


class TestSavePointsPreservesZ:
    """Test that saving 3D points preserves the z coordinate."""

    def test_save_3d_points_preserves_z(
        self,
        tmp_path: Path,
        make_napari_viewer: Any,
        sdata_3d_points: SpatialData,
    ):
        """Saving a 3D points layer must retain the z column in the stored element."""
        original_value = config.PROJECT_3D_POINTS_TO_2D
        try:
            config.PROJECT_3D_POINTS_TO_2D = False

            tmpdir = tmp_path / "sdata.zarr"
            sdata_3d_points.write(tmpdir)

            viewer = make_napari_viewer()
            widget = SdataWidget(viewer, EventedList([sdata_3d_points]))

            widget.coordinate_system_widget._select_coord_sys("global")
            widget.elements_widget._onItemChange("global")
            widget._onClick("points_3d")

            layer = viewer.layers[0]
            assert isinstance(layer, Points)
            assert layer.data.shape[1] == 3

            original_z = sdata_3d_points.points["points_3d"].compute()["z"].values.copy()

            parsed, _ = widget.viewer_model._save_points_to_sdata(layer, "points_3d", overwrite=True)

            saved_axes = get_axes_names(parsed)
            assert "z" in saved_axes, "z axis must be preserved after save"

            saved_z = parsed.compute()["z"].values
            np.testing.assert_array_almost_equal(saved_z, original_z)
        finally:
            config.PROJECT_3D_POINTS_TO_2D = original_value

    def test_save_2d_points_no_z(
        self,
        tmp_path: Path,
        make_napari_viewer: Any,
        sdata_3d_points: SpatialData,
    ):
        """When projected to 2D, saved points must not contain a z column."""
        original_value = config.PROJECT_3D_POINTS_TO_2D
        try:
            config.PROJECT_3D_POINTS_TO_2D = True

            tmpdir = tmp_path / "sdata.zarr"
            sdata_3d_points.write(tmpdir)

            viewer = make_napari_viewer()
            widget = SdataWidget(viewer, EventedList([sdata_3d_points]))

            widget.coordinate_system_widget._select_coord_sys("global")
            widget.elements_widget._onItemChange("global")
            widget._onClick("points_3d")

            layer = viewer.layers[0]
            assert isinstance(layer, Points)
            assert layer.data.shape[1] == 2

            parsed, _ = widget.viewer_model._save_points_to_sdata(layer, "points_3d", overwrite=True)

            saved_axes = get_axes_names(parsed)
            assert "z" not in saved_axes
        finally:
            config.PROJECT_3D_POINTS_TO_2D = original_value


class TestSaveShapesPreservesZ:
    """Test that saving 2.5D shapes preserves the z coordinate."""

    def test_save_2_5d_shapes_preserves_z(
        self,
        tmp_path: Path,
        make_napari_viewer: Any,
        sdata_2_5d_shapes: SpatialData,
    ):
        """Saving a 2.5D shapes layer must retain the z column in the stored element."""
        original_value = config.PROJECT_2_5D_SHAPES_TO_2D
        try:
            config.PROJECT_2_5D_SHAPES_TO_2D = False

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
                assert shape_data.shape[1] == 3

            parsed, _ = widget.viewer_model._save_shapes_to_sdata(layer, "shapes_2.5d", overwrite=True)

            saved_axes = get_axes_names(parsed)
            assert "z" in saved_axes, "z axis must be preserved after save"

            saved_z = parsed["z"].values
            original_unique_z = np.unique(sdata_2_5d_shapes.shapes["shapes_2.5d"]["z"].values)
            np.testing.assert_array_almost_equal(
                np.unique(saved_z), original_unique_z
            )
        finally:
            config.PROJECT_2_5D_SHAPES_TO_2D = original_value

    def test_save_2d_shapes_no_z(
        self,
        tmp_path: Path,
        make_napari_viewer: Any,
        sdata_2_5d_shapes: SpatialData,
    ):
        """When projected to 2D, saved shapes must not contain a z column."""
        original_value = config.PROJECT_2_5D_SHAPES_TO_2D
        try:
            config.PROJECT_2_5D_SHAPES_TO_2D = True

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
                assert shape_data.shape[1] == 2

            parsed, _ = widget.viewer_model._save_shapes_to_sdata(layer, "shapes_2.5d", overwrite=True)

            saved_axes = get_axes_names(parsed)
            assert "z" not in saved_axes
        finally:
            config.PROJECT_2_5D_SHAPES_TO_2D = original_value


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

            # Create a combined SpatialData
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
