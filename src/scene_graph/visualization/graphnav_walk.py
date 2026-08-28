"""Load a Boston Dynamics Spot GraphNav / Autowalk ``.walk`` map for visualisation.

ROS-free. ``bosdyn-api`` (pure-Python protobuf definitions — no Spot robot
connection, no C extensions) is imported lazily, so this module only costs the
dependency when a ``.walk`` map is actually opened.

A ``.walk`` directory (recorded by Autowalk on the tablet, or by the SDK's
``recording_command_line.py``) holds a serialized
``bosdyn.api.graph_nav.map_pb2.Graph`` protobuf plus ``waypoint_snapshots/`` and
``edge_snapshots/``. The ``graph`` file is looked for at the directory root and
under ``graph_nav/`` / ``downloaded_graph/`` (or you may pass the ``graph`` file
directly).

Global waypoint poses
---------------------
Waypoints store only *relative* transforms (``edge.from_tform_to``). Two ways to
lift them into a single frame:

* **anchoring** (preferred): ``graph.anchoring.anchors[wp] = seed_tform_waypoint``
  — a globally-consistent optimisation the map already carries when it was
  anchored (e.g. against a fiducial). ``graph.anchoring.objects`` additionally
  pins fiducials/world objects as ``seed_tform_object``.
* **bfs** (fallback): breadth-first from the first waypoint, composing
  ``from_tform_to`` along edges. The frame is then that root waypoint.

If your reconstruction is expressed in a fiducial frame, record/anchor the walk
against that same fiducial: the anchoring seed frame is then the fiducial frame,
so ``anchor="seed"`` (or ``"auto"``) needs no further transform. Use
``extra_transform`` for any residual 4x4 nudge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

_GRAPH_CANDIDATES = ("graph", "graph_nav/graph", "downloaded_graph/graph")


# ----------------------------------------------------------------------------
# small SE3 helpers (numpy only; bosdyn Quaternion is w, x, y, z)
# ----------------------------------------------------------------------------
def _quat_wxyz_to_rotmat(w: float, x: float, y: float, z: float) -> np.ndarray:
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def _rotmat_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    m = np.asarray(rot, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    nrm = float(np.linalg.norm(q))
    return q / nrm if nrm > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def _se3_to_matrix(se3) -> np.ndarray:
    """``bosdyn.api.SE3Pose`` protobuf -> 4x4 homogeneous matrix."""
    T = np.eye(4, dtype=np.float64)
    p, r = se3.position, se3.rotation
    T[:3, :3] = _quat_wxyz_to_rotmat(r.w, r.x, r.y, r.z)
    T[:3, 3] = (p.x, p.y, p.z)
    return T


# ----------------------------------------------------------------------------
# data model
# ----------------------------------------------------------------------------
@dataclass
class Waypoint:
    id: str
    name: str
    T_world: np.ndarray  # (4, 4) waypoint pose in the resolved world frame


@dataclass
class AnchoredObject:
    id: str
    T_world: np.ndarray  # (4, 4)


@dataclass
class NavGraph:
    waypoints: list[Waypoint] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
    anchored_objects: list[AnchoredObject] = field(default_factory=list)
    frame: str = "seed"  # "seed" | "bfs:<root_id>"
    source: str = ""

    # -- convenience accessors for the viser visualizer -----------------
    def waypoint_positions(self) -> np.ndarray:
        if not self.waypoints:
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack([w.T_world[:3, 3] for w in self.waypoints], axis=0).astype(np.float32)

    def waypoint_wxyz(self) -> np.ndarray:
        if not self.waypoints:
            return np.zeros((0, 4), dtype=np.float32)
        return np.stack([_rotmat_to_quat_wxyz(w.T_world[:3, :3]) for w in self.waypoints], axis=0).astype(np.float32)

    def waypoint_names(self) -> list[str]:
        return [w.name or w.id[:8] for w in self.waypoints]

    def edge_segments(self) -> np.ndarray:
        """(M, 2, 3) endpoint pairs for ``add_line_segments``."""
        by_id = {w.id: w.T_world[:3, 3] for w in self.waypoints}
        segs = [
            np.stack([by_id[a], by_id[b]], axis=0)
            for a, b in self.edges
            if a in by_id and b in by_id
        ]
        if not segs:
            return np.zeros((0, 2, 3), dtype=np.float32)
        return np.stack(segs, axis=0).astype(np.float32)

    def anchored_object_positions(self) -> np.ndarray:
        if not self.anchored_objects:
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack([o.T_world[:3, 3] for o in self.anchored_objects], axis=0).astype(np.float32)


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
def _resolve_graph_file(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        for rel in _GRAPH_CANDIDATES:
            cand = path / rel
            if cand.is_file():
                return cand
        # last resort: a single file literally named "graph" anywhere shallow
        hits = sorted(path.glob("**/graph"))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"No GraphNav 'graph' protobuf found at {path} "
        f"(looked for {', '.join(_GRAPH_CANDIDATES)})"
    )


def _load_graph_proto(graph_file: Path):
    try:
        from bosdyn.api.graph_nav import map_pb2  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency guidance
        raise ImportError(
            "Reading a Spot .walk map needs the 'bosdyn-api' package "
            "(pip install bosdyn-api). It is protobuf-only — no robot connection."
        ) from exc
    graph = map_pb2.Graph()
    graph.ParseFromString(graph_file.read_bytes())
    return graph


def _bfs_world_poses(graph, root_id: str | None) -> tuple[dict[str, np.ndarray], str]:
    """Compose ``edge.from_tform_to`` breadth-first into one frame."""
    wp_ids = [w.id for w in graph.waypoints]
    if not wp_ids:
        return {}, ""
    root = root_id if (root_id in wp_ids) else wp_ids[0]

    adj: dict[str, list[tuple[str, np.ndarray, bool]]] = {i: [] for i in wp_ids}
    for e in graph.edges:
        a, b = e.id.from_waypoint, e.id.to_waypoint
        if a not in adj or b not in adj:
            continue
        T_a_b = _se3_to_matrix(e.from_tform_to)
        adj[a].append((b, T_a_b, True))
        adj[b].append((a, np.linalg.inv(T_a_b), False))

    poses: dict[str, np.ndarray] = {}
    remaining = set(wp_ids)
    seeds = [root] + [i for i in wp_ids if i != root]
    components = 0
    for seed in seeds:
        if seed not in remaining:
            continue
        components += 1
        poses[seed] = np.eye(4, dtype=np.float64)
        remaining.discard(seed)
        queue = [seed]
        while queue:
            cur = queue.pop(0)
            for nb, T_rel, _fwd in adj[cur]:
                if nb in poses:
                    continue
                poses[nb] = poses[cur] @ T_rel
                remaining.discard(nb)
                queue.append(nb)
        if not remaining:
            break
    if components > 1:
        LOGGER.warning(
            "graphnav_walk: BFS found %d disconnected components across %d "
            "waypoints; components after the first are re-seeded at the origin "
            "and will overlap. Anchor the map (--walk-anchor seed) for a "
            "globally-consistent layout.",
            components, len(wp_ids),
        )
    return poses, f"bfs:{root}"


def load_walk_graph(
    walk_dir: str | Path,
    *,
    anchor: str = "auto",
    extra_transform: np.ndarray | None = None,
    root_waypoint: str | None = None,
) -> NavGraph:
    """Parse a Spot ``.walk`` map into a :class:`NavGraph` of world-frame poses.

    Parameters
    ----------
    walk_dir:
        The ``.walk`` directory (or the ``graph`` file directly).
    anchor:
        ``"auto"`` — use ``graph.anchoring`` if present, else BFS.
        ``"seed"`` — require anchoring (error if the map is not anchored).
        ``"bfs"``  — always compose edge transforms from ``root_waypoint``.
    extra_transform:
        Optional row-major 4x4 applied to every pose: ``T_world = extra @ T_seed``.
    root_waypoint:
        BFS root waypoint id (default: the graph's first waypoint).
    """
    graph_file = _resolve_graph_file(Path(walk_dir))
    graph = _load_graph_proto(graph_file)

    extra = None
    if extra_transform is not None:
        extra = np.asarray(extra_transform, dtype=np.float64).reshape(4, 4)

    anchoring = getattr(graph, "anchoring", None)
    has_anchors = bool(anchoring is not None and len(anchoring.anchors) > 0)
    mode = anchor.lower().strip()
    if mode == "seed" and not has_anchors:
        raise ValueError(
            f"{graph_file} has no anchoring; re-run anchoring on the map or use "
            "anchor='bfs'."
        )
    use_anchoring = has_anchors if mode in ("auto", "seed") else False

    name_by_id = {w.id: (w.annotations.name or "") for w in graph.waypoints}
    world_by_id: dict[str, np.ndarray] = {}
    anchored_objects: list[AnchoredObject] = []

    if use_anchoring:
        frame = "seed"
        for a in anchoring.anchors:
            world_by_id[a.id] = _se3_to_matrix(a.seed_tform_waypoint)
        for obj in getattr(anchoring, "objects", []):
            T = _se3_to_matrix(obj.seed_tform_object)
            if extra is not None:
                T = extra @ T
            anchored_objects.append(AnchoredObject(id=obj.id, T_world=T))
    else:
        world_by_id, frame = _bfs_world_poses(graph, root_waypoint)

    waypoints: list[Waypoint] = []
    for w in graph.waypoints:
        T = world_by_id.get(w.id)
        if T is None:
            continue
        if extra is not None:
            T = extra @ T
        waypoints.append(
            Waypoint(id=w.id, name=name_by_id.get(w.id, ""), T_world=np.asarray(T, dtype=np.float64))
        )

    edges = [
        (e.id.from_waypoint, e.id.to_waypoint)
        for e in graph.edges
        if e.id.from_waypoint and e.id.to_waypoint
    ]

    nav = NavGraph(
        waypoints=waypoints,
        edges=edges,
        anchored_objects=anchored_objects,
        frame=frame,
        source=str(graph_file),
    )
    LOGGER.info(
        "graphnav_walk: loaded %d waypoints, %d edges, %d anchored objects from %s (frame=%s)",
        len(nav.waypoints), len(nav.edge_segments()), len(nav.anchored_objects), graph_file, frame,
    )
    return nav
