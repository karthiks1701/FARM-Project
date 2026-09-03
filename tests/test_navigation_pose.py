"""Tests for the collision-aware navigation-pose resolver."""

from __future__ import annotations

import math

import numpy as np

from scene_graph.retrieval.navigation_pose import (
    NavigationPose,
    compute_navigation_pose,
    decode_object_voxel_points,
    navigation_poses_for_scene,
)


def _wall_points(x: float, y0: float, y1: float, z: float = 0.0, n: int = 120) -> np.ndarray:
    ys = np.linspace(y0, y1, n)
    return np.stack([np.full(n, x), ys, np.full(n, z)], axis=1)


def test_pose_clears_robot_radius_plus_barrier():
    # Target at origin; one obstacle object forms a wall at x = 0.6.
    means = np.array([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]], dtype=np.float64)
    obs = _wall_points(0.6, -3.0, 3.0)
    pts = np.concatenate([np.zeros((1, 3)), obs], axis=0)
    owner = np.concatenate([np.zeros(1, np.int64), np.ones(len(obs), np.int64)])

    nav = compute_navigation_pose(
        0, means=means, voxel_points=pts, voxel_owner=owner,
        clearance_margin_m=0.10, robot_radius_m=0.5, search_radius_m=3.0, n_angles=72,
    )
    assert isinstance(nav, NavigationPose)
    assert nav.navigable
    assert nav.required_clearance_m == 0.6
    assert nav.clearance_m >= 0.6 - 1e-6
    # Must not be placed on the obstacle side (x should be <= wall - required).
    assert nav.position[0] <= 0.6 - 0.6 + 1e-6
    # Heading points back at the target (origin).
    dx = math.cos(nav.yaw_rad)
    dy = math.sin(nav.yaw_rad)
    to_target = np.array([0.0, 0.0]) - np.array(nav.position[:2])
    to_target /= np.linalg.norm(to_target)
    assert np.dot([dx, dy], to_target) > 0.95


def test_flags_unnavigable_when_boxed_in():
    # Target buried in a thick annulus of obstacle clutter (r 0.3..1.6 m) on all
    # sides -> a 0.5 m robot has no body-safe pose within the search radius; a
    # pose is still returned, flagged navigable=False.
    means = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    rng = np.random.default_rng(1)
    n = 4000
    rad = rng.uniform(0.3, 1.6, size=n)
    ang = rng.uniform(0, 2 * math.pi, size=n)
    clutter = np.stack([rad * np.cos(ang), rad * np.sin(ang), np.zeros(n)], axis=1)
    owner = np.ones(n, np.int64)  # all belong to object 1
    nav = compute_navigation_pose(
        0, means=means, voxel_points=clutter, voxel_owner=owner,
        clearance_margin_m=0.10, robot_radius_m=0.5, search_radius_m=0.8,
    )
    assert not nav.navigable
    assert nav.clearance_m < nav.required_clearance_m
    assert "NO body-safe pose" in nav.note


def test_no_obstacles_returns_infinite_clearance():
    means = np.array([[1.0, 2.0, 0.5]], dtype=np.float64)
    nav = compute_navigation_pose(
        0, means=means,
        voxel_points=np.zeros((0, 3)), voxel_owner=np.zeros((0,), np.int64),
        robot_radius_m=0.5,
    )
    assert nav.navigable
    assert math.isinf(nav.clearance_m)
    assert nav.target_position == (1.0, 2.0, 0.5)
    # vertical component preserved from the target centroid
    assert abs(nav.position[2] - 0.5) < 1e-9


def test_workspace_bounds_keep_pose_off_the_walls():
    # Target hard against the x=0 wall, no mapped obstacles. Without bounds the
    # ring search would happily stand at x<0 (inside/through the wall).
    means = np.array([[0.3, 3.0, 0.0]], dtype=np.float64)
    empty, own = np.zeros((0, 3)), np.zeros((0,), np.int64)

    free = compute_navigation_pose(
        0, means=means, voxel_points=empty, voxel_owner=own,
        robot_radius_m=0.6, clearance_margin_m=0.1,
    )
    bounded = compute_navigation_pose(
        0, means=means, voxel_points=empty, voxel_owner=own,
        robot_radius_m=0.6, clearance_margin_m=0.1,
        workspace_bounds=(0.0, 8.0, None, None),
    )
    assert bounded.navigable
    # never negative, and a full robot_radius+margin (0.70 m) off the x=0 wall
    assert bounded.position[0] >= 0.70 - 1e-6
    assert bounded.position[0] <= 8.0
    # a bound may also be reported as the limiting factor
    assert "wall" in bounded.note or bounded.position[0] >= 0.70


def test_workspace_bounds_flag_unnavigable_when_pinned():
    # Target pinned between the x=0 wall and a shelf at x in [0.5, 6];
    # with a short search there is no room for a 0.6 m robot -> not navigable,
    # and the best-effort pose still respects the bounds.
    means = np.array([[0.2, 3.0, 0.0], [3.0, 3.0, 0.0]], dtype=np.float64)
    gx, gy = np.meshgrid(np.linspace(0.5, 6.0, 300), np.linspace(0.0, 6.0, 30))
    obs = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    own = np.ones(obs.shape[0], np.int64)
    nav = compute_navigation_pose(
        0, means=means, voxel_points=obs, voxel_owner=own,
        robot_radius_m=0.6, clearance_margin_m=0.1,
        workspace_bounds=(0.0, 8.0, None, None), search_radius_m=1.0,
    )
    assert not nav.navigable
    assert 0.0 <= nav.position[0] <= 8.0
    assert "workspace bounds" in nav.note


def test_stands_close_to_an_elongated_object_not_its_bounding_radius():
    # A ~1.6 m long "hose" centered at (7.6, -0.97); a rack of other-object
    # voxels well away. The stand pose should hug the hose (~target_standoff
    # from its NEAREST part), not sit a whole hose-length out from the centroid.
    t = np.linspace(0, 3 * math.pi, 240)
    r = 0.3 + 0.15 * t / 9.0
    hose = np.stack([7.6 + 0.7 * np.cos(t) * r, -0.97 + 0.7 * np.sin(t) * r, np.zeros_like(t)], axis=1)
    rack = np.stack([np.full(80, 6.0), np.linspace(-3, 0.5, 80), np.zeros(80)], axis=1)
    pts = np.concatenate([hose, rack], axis=0)
    own = np.concatenate([np.zeros(len(hose), int), np.ones(len(rack), int)])
    means = np.array([[7.6, -0.97, 0.0], [6.0, -1.0, 0.0]], dtype=np.float64)

    nav = compute_navigation_pose(
        0, means=means, voxel_points=pts, voxel_owner=own,
        robot_radius_m=0.6, clearance_margin_m=0.1, target_standoff_m=0.8,
        workspace_bounds=(0.0, 8.0, None, None),
    )
    assert nav.navigable
    # ~0.8 m from the nearest hose voxel, and comfortably closer than a full
    # hose-length (the old ring-from-centroid behaviour put it ~3 m away).
    assert abs(nav.offset_from_target_m - 0.8) < 0.35
    cx, cy = nav.target_position[0], nav.target_position[1]
    dist_to_centroid = math.hypot(nav.position[0] - cx, nav.position[1] - cy)
    assert dist_to_centroid < 1.8
    assert 0.0 <= nav.position[0] <= 8.0


def test_self_voxels_never_count_as_obstacles():
    # Big target blob + a far obstacle: clearance is measured to the obstacle,
    # not to the target's own voxels, so the pose sits just outside the blob.
    rng = np.random.default_rng(0)
    blob = rng.uniform(-0.4, 0.4, size=(300, 3))
    blob[:, 2] = 0.0
    far = _wall_points(5.0, -2.0, 2.0)
    pts = np.concatenate([blob, far], axis=0)
    owner = np.concatenate([np.zeros(len(blob), np.int64), np.ones(len(far), np.int64)])
    means = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float64)
    nav = compute_navigation_pose(
        0, means=means, voxel_points=pts, voxel_owner=owner,
        robot_radius_m=0.3, clearance_margin_m=0.1, search_radius_m=2.0,
    )
    assert nav.navigable
    # Close to the target (just outside its ~0.55 m reach), far from the wall.
    assert nav.offset_from_target_m < 1.5
    assert nav.clearance_m > 1.0


def test_decode_and_scene_helper_smoke():
    # No voxel buffers -> empty decode, helper still returns a pose per index
    # (infinite clearance, since there are no obstacles).
    state = {"means": np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64)}
    pts, owner = decode_object_voxel_points(state)
    assert pts.shape == (0, 3) and owner.shape == (0,)
    out = navigation_poses_for_scene(state, [0, 1], robot_radius_m=0.5)
    assert set(out) == {0, 1}
    assert all(p.navigable for p in out.values())
