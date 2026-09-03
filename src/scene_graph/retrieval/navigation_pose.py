"""Collision-aware navigation pose for a retrieved object.

A relational query answers *which object* — its natural "pose" is the object
**centroid**. But that point sits inside the object; handing it to the robot as a
navigation goal invites a collision with the target or its neighbours. This
module derives a nearby standoff pose the robot body can actually occupy:
horizontally outside every *other* object's voxel evidence by at least

    required_clearance = robot_radius_m + clearance_margin_m

where ``clearance_margin_m`` is a small safety barrier (a few centimetres) and
``robot_radius_m`` is the robot footprint half-width (Spot ~0.6 m). The search is
a widening ring of candidate stand points around the target on the horizontal
plane; the nearest ring slot that clears the requirement wins, and the heading
faces the target. If nothing within ``search_radius_m`` clears the body+barrier
requirement the best-effort pose is still returned with ``navigable=False`` and a
note, so callers never silently ship an un-navigable goal.

ROS-free; numpy + scipy (cKDTree) only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from scene_graph.utils.geometry import decode_voxel_keys_numpy

_DEFAULT_CLEARANCE_MARGIN_M = 0.10
_DEFAULT_ROBOT_RADIUS_M = 0.6


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


@dataclass
class NavigationPose:
    """A body-safe navigation goal near a retrieved object.

    ``position`` is the robot body-centre goal in world coordinates; its
    vertical component is copied from the target centroid (obstacles here are
    object voxels, not the floor) — a downstream planner should project it onto
    the navigation surface. ``yaw_rad`` faces the target.
    """

    position: tuple[float, float, float]
    yaw_rad: float
    clearance_m: float
    required_clearance_m: float
    offset_from_target_m: float
    target_position: tuple[float, float, float]
    navigable: bool
    note: str
    robot_radius_m: float = _DEFAULT_ROBOT_RADIUS_M
    clearance_margin_m: float = _DEFAULT_CLEARANCE_MARGIN_M

    def as_dict(self) -> dict:
        return {
            "position": [float(v) for v in self.position],
            "yaw_rad": float(self.yaw_rad),
            "yaw_deg": float(math.degrees(self.yaw_rad)),
            "clearance_m": float(self.clearance_m),
            "required_clearance_m": float(self.required_clearance_m),
            "offset_from_target_m": float(self.offset_from_target_m),
            "target_position": [float(v) for v in self.target_position],
            "navigable": bool(self.navigable),
            "note": str(self.note),
            "robot_radius_m": float(self.robot_radius_m),
            "clearance_margin_m": float(self.clearance_margin_m),
        }

    def summary(self) -> str:
        px, py, pz = self.position
        mark = "ok" if self.navigable else "UNSAFE"
        return (
            f"nav_pose=({px:.2f}, {py:.2f}, {pz:.2f}) yaw={math.degrees(self.yaw_rad):.0f}° "
            f"clearance={self.clearance_m:.2f}m/{self.required_clearance_m:.2f}m "
            f"offset={self.offset_from_target_m:.2f}m [{mark}]"
        )


def decode_object_voxel_points(state: dict) -> tuple[np.ndarray, np.ndarray]:
    """Decode every object's sparse voxel cloud to world points.

    Returns ``(points (P, 3) float32, owner_index (P,) int64)`` where
    ``owner_index`` is the object row each point belongs to. Empty arrays if the
    scene state carries no voxel buffers.
    """
    flat_t = state.get("object_voxel_keys_flat")
    offs_t = state.get("object_voxel_keys_offsets")
    lvl_t = state.get("object_voxel_levels")
    if flat_t is None or offs_t is None or lvl_t is None:
        return np.zeros((0, 3), np.float32), np.zeros((0,), np.int64)
    flat = _to_numpy(flat_t).astype(np.int64, copy=False).reshape(-1)
    offsets = _to_numpy(offs_t).astype(np.int64, copy=False).reshape(-1)
    levels = _to_numpy(lvl_t).astype(np.int64, copy=False).reshape(-1)
    if offsets.shape[0] < 2 or flat.shape[0] == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0,), np.int64)

    active = state.get("active")
    active_np = None
    if active is not None:
        try:
            active_np = _to_numpy(active).astype(bool).reshape(-1)
        except Exception:
            active_np = None

    pts_chunks: list[np.ndarray] = []
    owner_chunks: list[np.ndarray] = []
    n_obj = offsets.shape[0] - 1
    for i in range(n_obj):
        if active_np is not None and i < active_np.shape[0] and not bool(active_np[i]):
            continue
        start, end = int(offsets[i]), int(offsets[i + 1])
        if end <= start:
            continue
        level = int(levels[i]) if i < levels.shape[0] else 0
        pts = decode_voxel_keys_numpy(flat[start:end], level)
        if pts.shape[0] == 0:
            continue
        pts_chunks.append(pts.astype(np.float32, copy=False))
        owner_chunks.append(np.full((pts.shape[0],), i, dtype=np.int64))
    if not pts_chunks:
        return np.zeros((0, 3), np.float32), np.zeros((0,), np.int64)
    return np.concatenate(pts_chunks, axis=0), np.concatenate(owner_chunks, axis=0)


def compute_navigation_pose(
    target_index: int,
    *,
    means: np.ndarray,
    voxel_points: np.ndarray,
    voxel_owner: np.ndarray,
    clearance_margin_m: float = _DEFAULT_CLEARANCE_MARGIN_M,
    robot_radius_m: float = _DEFAULT_ROBOT_RADIUS_M,
    up_axis: int = 2,
    search_radius_m: float = 2.5,
    radial_step_m: float = 0.1,
    n_angles: int = 48,
    vertical_band_m: float = 1.5,
    workspace_bounds: tuple | None = None,
) -> NavigationPose:
    """Body-safe standoff pose near object ``target_index``.

    ``means`` is ``(N, 3)`` object centroids. ``voxel_points`` / ``voxel_owner``
    come from :func:`decode_object_voxel_points`. Points owned by
    ``target_index`` are excluded from the clearance metric (we *want* to stand
    near the target); every other object's voxels are obstacles.

    ``workspace_bounds`` — optional ``(xmin, xmax, ymin, ymax)`` in the two
    horizontal axes (any entry may be ``None``). Known room walls the map does
    NOT contain as objects: a candidate stand point is required to keep the full
    ``robot_radius + margin`` from every set bound too, so a pose near a boundary
    object is never placed into / through the wall.
    """
    means = np.asarray(means, dtype=np.float64)
    c = means[int(target_index)].astype(np.float64)
    up = int(up_axis) % 3
    horiz = [a for a in range(3) if a != up]
    c2 = c[horiz]
    required = float(robot_radius_m) + float(clearance_margin_m)

    xmin = xmax = ymin = ymax = None
    if workspace_bounds is not None:
        try:
            xmin, xmax, ymin, ymax = (
                None if v is None else float(v) for v in tuple(workspace_bounds)[:4]
            )
        except Exception:
            xmin = xmax = ymin = ymax = None

    def _bound_clearance(cands: np.ndarray) -> np.ndarray:
        """Signed inward distance to the nearest set workspace bound (+inf if none)."""
        out = np.full(cands.shape[0], np.inf, dtype=np.float64)
        if xmin is not None:
            out = np.minimum(out, cands[:, 0] - xmin)
        if xmax is not None:
            out = np.minimum(out, xmax - cands[:, 0])
        if ymin is not None:
            out = np.minimum(out, cands[:, 1] - ymin)
        if ymax is not None:
            out = np.minimum(out, ymax - cands[:, 1])
        return out

    pts = np.asarray(voxel_points, dtype=np.float64).reshape(-1, 3)
    owner = np.asarray(voxel_owner, dtype=np.int64).reshape(-1)
    if pts.shape[0] != owner.shape[0]:
        pts = np.zeros((0, 3)); owner = np.zeros((0,), np.int64)

    self_mask = owner == int(target_index)
    obs_mask = ~self_mask
    if vertical_band_m and vertical_band_m > 0.0 and pts.shape[0]:
        obs_mask &= np.abs(pts[:, up] - c[up]) <= float(vertical_band_m)
    obs2 = pts[obs_mask][:, horiz]

    # Start the ring just outside the target's own horizontal footprint.
    self2 = pts[self_mask][:, horiz]
    if self2.shape[0]:
        target_reach = float(np.linalg.norm(self2 - c2[None, :], axis=1).max())
    else:
        target_reach = 0.0
    r_start = max(required, target_reach + 0.05)

    def _pose(pos2: np.ndarray, clearance: float, navigable: bool, note: str) -> NavigationPose:
        pos2 = np.asarray(pos2, dtype=np.float64).copy()
        # Hard guarantee: the reported pose is never outside the workspace box,
        # even on the best-effort (navigable=False) path.
        clamped = False
        if xmin is not None and pos2[0] < xmin:
            pos2[0], clamped = xmin, True
        if xmax is not None and pos2[0] > xmax:
            pos2[0], clamped = xmax, True
        if ymin is not None and pos2[1] < ymin:
            pos2[1], clamped = ymin, True
        if ymax is not None and pos2[1] > ymax:
            pos2[1], clamped = ymax, True
        if clamped:
            note = note + " (clamped to workspace bounds)"
        p3 = c.copy()
        p3[horiz[0]] = float(pos2[0])
        p3[horiz[1]] = float(pos2[1])
        delta = c2 - pos2
        yaw = math.atan2(float(delta[1]), float(delta[0]))
        return NavigationPose(
            position=(float(p3[0]), float(p3[1]), float(p3[2])),
            yaw_rad=float(yaw),
            clearance_m=float(clearance),
            required_clearance_m=float(required),
            offset_from_target_m=float(np.linalg.norm(pos2 - c2)),
            target_position=(float(c[0]), float(c[1]), float(c[2])),
            navigable=bool(navigable),
            note=note,
            robot_radius_m=float(robot_radius_m),
            clearance_margin_m=float(clearance_margin_m),
        )

    angles = np.linspace(0.0, 2.0 * math.pi, int(max(4, n_angles)), endpoint=False)
    ring = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (A, 2)

    if obs2.shape[0] == 0:
        def _voxel_clearance(cands: np.ndarray) -> np.ndarray:
            return np.full(cands.shape[0], np.inf, dtype=np.float64)
    else:
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(obs2)

            def _voxel_clearance(cands: np.ndarray) -> np.ndarray:
                d, _ = tree.query(cands, k=1)
                return np.asarray(d, dtype=np.float64)
        except Exception:
            def _voxel_clearance(cands: np.ndarray) -> np.ndarray:
                diff = cands[:, None, :] - obs2[None, :, :]
                return np.sqrt((diff * diff).sum(-1)).min(axis=1)

    def _clearance(cands: np.ndarray) -> np.ndarray:
        # Effective clearance = nearest obstacle voxel OR nearest workspace wall.
        return np.minimum(_voxel_clearance(cands), _bound_clearance(cands))

    radii = np.arange(r_start, r_start + float(search_radius_m) + 1e-6, float(radial_step_m))
    best_pos2: np.ndarray | None = None
    best_clear = -1e9
    for r in radii:
        cands = c2[None, :] + r * ring  # (A, 2)
        clr = _clearance(cands)
        j = int(np.argmax(clr))
        if clr[j] > best_clear:
            best_clear, best_pos2 = float(clr[j]), cands[j]
        ok = clr >= required
        if np.any(ok):
            idx = np.where(ok)[0]
            k = idx[int(np.argmax(clr[idx]))]
            pos2 = cands[k]
            lim = "wall" if _bound_clearance(cands[k:k + 1])[0] <= _voxel_clearance(cands[k:k + 1])[0] else "object"
            return _pose(
                pos2, float(clr[k]), True,
                f"standoff {r:.2f} m from target; {float(clr[k]) - required:+.2f} m past the "
                f"{required:.2f} m body+barrier requirement (nearest limit: {lim})",
            )

    assert best_pos2 is not None
    bounded = workspace_bounds is not None
    return _pose(
        best_pos2, best_clear, False,
        f"NO body-safe pose within {search_radius_m:.1f} m of the target"
        f"{' inside the workspace bounds' if bounded else ''}: best clearance "
        f"{best_clear:.2f} m < required {required:.2f} m. Approach manually, shrink the "
        f"robot footprint, widen the search, or relax the bounds.",
    )


def navigation_poses_for_scene(
    state: dict,
    target_indices: Iterable[int],
    **kwargs,
) -> dict[int, NavigationPose]:
    """Convenience: decode the scene once, return ``{object_index: NavigationPose}``.

    ``kwargs`` pass straight to :func:`compute_navigation_pose` (``clearance_margin_m``,
    ``robot_radius_m``, ``up_axis``, ``search_radius_m``, ...).
    """
    means = _to_numpy(state.get("means")).astype(np.float64)
    pts, owner = decode_object_voxel_points(state)
    out: dict[int, NavigationPose] = {}
    for i in target_indices:
        i = int(i)
        if 0 <= i < means.shape[0]:
            out[i] = compute_navigation_pose(
                i, means=means, voxel_points=pts, voxel_owner=owner, **kwargs
            )
    return out
