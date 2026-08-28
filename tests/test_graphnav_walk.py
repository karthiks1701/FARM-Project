"""Unit tests for the Spot GraphNav ``.walk`` loader helpers.

These cover the ROS-free / bosdyn-free math (SE3 <-> matrix, BFS pose
composition, :class:`NavGraph` accessors). The full ``load_walk_graph`` path
needs ``bosdyn-api`` to parse a real ``graph`` protobuf and is exercised by a
short offline viewer run instead.
"""

from __future__ import annotations

import numpy as np

from scene_graph.visualization.graphnav_walk import (
    NavGraph,
    Waypoint,
    _bfs_world_poses,
    _quat_wxyz_to_rotmat,
    _rotmat_to_quat_wxyz,
    _se3_to_matrix,
)


class _Vec:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _Quat:
    def __init__(self, w, x, y, z):
        self.w, self.x, self.y, self.z = w, x, y, z


class _SE3:
    def __init__(self, pos, quat=(1.0, 0.0, 0.0, 0.0)):
        self.position = _Vec(*pos)
        self.rotation = _Quat(*quat)


class _WP:
    def __init__(self, wp_id):
        self.id = wp_id


class _EdgeId:
    def __init__(self, a, b):
        self.from_waypoint, self.to_waypoint = a, b


class _Edge:
    def __init__(self, a, b, se3):
        self.id = _EdgeId(a, b)
        self.from_tform_to = se3


class _Graph:
    def __init__(self, waypoints, edges):
        self.waypoints = waypoints
        self.edges = edges


def test_quat_matrix_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)
        rot = _quat_wxyz_to_rotmat(*q)
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-9)
        rot2 = _quat_wxyz_to_rotmat(*_rotmat_to_quat_wxyz(rot))
        assert np.allclose(rot, rot2, atol=1e-8)


def test_se3_to_matrix_identity_rotation():
    mat = _se3_to_matrix(_SE3((1.0, 2.0, 3.0)))
    assert np.allclose(mat[:3, :3], np.eye(3))
    assert np.allclose(mat[:3, 3], [1.0, 2.0, 3.0])


def test_bfs_composes_edge_transforms_along_a_chain():
    # a --(+1x)--> b --(+1y)--> c ; root at a -> c should land at (1, 1, 0).
    graph = _Graph(
        waypoints=[_WP("a"), _WP("b"), _WP("c")],
        edges=[
            _Edge("a", "b", _SE3((1.0, 0.0, 0.0))),
            _Edge("b", "c", _SE3((0.0, 1.0, 0.0))),
        ],
    )
    poses, frame = _bfs_world_poses(graph, "a")
    assert frame == "bfs:a"
    assert np.allclose(poses["a"][:3, 3], [0.0, 0.0, 0.0])
    assert np.allclose(poses["b"][:3, 3], [1.0, 0.0, 0.0])
    assert np.allclose(poses["c"][:3, 3], [1.0, 1.0, 0.0])


def test_bfs_traverses_edges_backwards():
    # Only edge is a->b; rooting at b must still place a via the inverse.
    graph = _Graph(
        waypoints=[_WP("a"), _WP("b")],
        edges=[_Edge("a", "b", _SE3((2.0, 0.0, 0.0)))],
    )
    poses, _ = _bfs_world_poses(graph, "b")
    assert np.allclose(poses["b"][:3, 3], [0.0, 0.0, 0.0])
    assert np.allclose(poses["a"][:3, 3], [-2.0, 0.0, 0.0])


def test_navgraph_accessors_skip_unknown_edge_endpoints():
    t_a = np.eye(4)
    t_b = np.eye(4)
    t_b[:3, 3] = [3.0, 0.0, 0.0]
    nav = NavGraph(
        waypoints=[Waypoint("a", "start", t_a), Waypoint("b", "", t_b)],
        edges=[("a", "b"), ("b", "ghost")],
    )
    assert nav.waypoint_positions().shape == (2, 3)
    assert nav.waypoint_wxyz().shape == (2, 4)
    assert nav.waypoint_names() == ["start", "b"]
    segs = nav.edge_segments()
    assert segs.shape == (1, 2, 3)  # ("b", "ghost") dropped
    assert np.allclose(segs[0], [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
