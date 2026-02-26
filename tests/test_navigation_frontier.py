import numpy as np

from my_project.navigation import FREE, OCCUPIED, UNKNOWN, FrontierPlanner, GridBounds, OccupancyGrid


def _cell_state(grid: OccupancyGrid, x: float, y: float) -> int | None:
    idx = grid.world_to_grid([x, y])
    if idx is None:
        return None
    return int(grid.grid[idx])


def test_occupancy_grid_update_marks_free_hit_and_unknown():
    grid = OccupancyGrid(
        resolution=0.1,
        bounds=GridBounds(x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0),
        ray_length=1.0,
        hit_epsilon=0.02,
    )
    drone = np.array([0.0, 0.0, 1.0], dtype=float)
    ray_dists = np.array([0.5, 1.0, 0.7, 1.0], dtype=float)
    ray_dirs_world = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    )

    grid.update(drone_pos=drone, ray_dists=ray_dists, ray_dirs_world=ray_dirs_world)

    assert _cell_state(grid, 0.0, 0.0) == FREE
    assert _cell_state(grid, 0.5, 0.0) == OCCUPIED
    assert _cell_state(grid, 0.3, 0.0) == FREE
    assert _cell_state(grid, 0.8, 0.0) == UNKNOWN
    assert _cell_state(grid, 0.0, 0.7) == OCCUPIED
    assert len(grid.get_frontier_indices()) > 0


def test_occupancy_grid_default_bounds_cover_negative_coords():
    grid = OccupancyGrid()
    assert grid.world_to_grid([-3.4, 2.6]) is not None
    assert grid.world_to_grid([4.6, -2.8]) is not None


def test_frontier_planner_returns_nearest_waypoint():
    grid = OccupancyGrid(
        resolution=0.1,
        bounds=GridBounds(x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0),
        ray_length=1.0,
    )
    planner = FrontierPlanner(grid, min_cluster_size=1, waypoint_z=1.0)

    drone = np.array([0.0, 0.0, 1.0], dtype=float)
    ray_dists = np.array([0.5, 1.0, 0.5, 1.0], dtype=float)
    ray_dirs_world = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=float,
    )

    planner.update(drone_pos=drone, ray_dists=ray_dists, ray_dirs_world=ray_dirs_world)
    wp = planner.get_next_waypoint(drone)

    assert wp is not None
    assert wp.shape == (3,)
    assert grid.is_free(wp[:2])
    assert not planner.is_exploration_done()


def test_frontier_planner_done_after_no_frontier_streak():
    grid = OccupancyGrid(
        resolution=0.2,
        bounds=GridBounds(x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5),
        ray_length=1.0,
    )
    planner = FrontierPlanner(grid, done_frontier_streak=2)

    # No unknown cells -> no frontiers.
    grid.grid.fill(FREE)

    drone = np.array([0.0, 0.0, 1.0], dtype=float)
    empty_dists = np.zeros((0,), dtype=float)
    empty_dirs = np.zeros((0, 3), dtype=float)

    planner.update(drone_pos=drone, ray_dists=empty_dists, ray_dirs_world=empty_dirs)
    assert planner.get_next_waypoint(drone) is None
    assert not planner.is_exploration_done()

    planner.update(drone_pos=drone, ray_dists=empty_dists, ray_dirs_world=empty_dirs)
    assert planner.is_exploration_done()
