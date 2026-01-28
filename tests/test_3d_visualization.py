"""Tests for 3D points and 2.5D shapes visualization.

For debugging tips on how to visually inspect tests, see docs/contributing.md.
"""

from typing import Any

import pytest
from napari.layers import Points, Shapes
from napari.utils.events import EventedList
from spatialdata import SpatialData

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
