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
    search_radius_m: float = 3.0,
    radial_step_m: float = 0.12,
    n_angles: int = 48,  # kept for API compat; unused by the grid search
    vertical_band_m: float = 1.5,
    workspace_bounds: tuple | None = None,
    target_standoff_m: float = 0.8,
    min_target_standoff_m: float = 0.3,
    max_target_dist_m: float = 2.0,
    _shared_obstacles: tuple | None = None,
) -> NavigationPose:
    """Body-safe stand pose **as close as possible** to object ``target_index``.

    ``means`` is ``(N, 3)`` object centroids. ``voxel_points`` / ``voxel_owner``
    come from :func:`decode_object_voxel_points`.

    Grid search over the horizontal plane: a candidate is feasible when it clears
    every *other* object's voxels and every workspace wall by
    ``robot_radius_m + clearance_margin_m``, and is at least
    ``min_target_standoff_m`` from the target's *own* voxels (not standing on
    it). Among feasible candidates the one whose distance to the nearest target
    voxel is closest to ``target_standoff_m`` wins — so the pose hugs the object
    (its nearest reachable part, which matters for long / coiled objects like a
    hose), not its bounding radius. ``navigable`` is False when the best feasible
    pose is further than ``max_target_dist_m`` from the object (or none exists).

    ``workspace_bounds`` — optional ``(xmin, xmax, ymin, ymax)``; any entry may
    be ``None``. Known room walls the map has no objects for.
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
    self2 = pts[self_mask][:, horiz]
    # Obstacles = every OTHER object's voxels within the vertical band. When a
    # pre-built tree over ALL voxels is supplied (batch path), reuse it and mask
    # out this target by owner at query time instead of rebuilding.
    shared_tree = shared_owner = None
    if _shared_obstacles is not None:
        shared_tree, shared_owner = _shared_obstacles[0], _shared_obstacles[1]
    obs2 = np.zeros((0, 2))
    if shared_tree is None:
        obs_mask = ~self_mask
        if vertical_band_m and vertical_band_m > 0.0 and pts.shape[0]:
            obs_mask &= np.abs(pts[:, up] - c[up]) <= float(vertical_band_m)
        obs2 = pts[obs_mask][:, horiz]

    def _pose(pos2: np.ndarray, clearance: float, navigable: bool, note: str, *, target_dist: float | None = None) -> NavigationPose:
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
        # Heading: face the target — its nearest own voxel if we have one,
        # else the centroid (better for long objects than always the centroid).
        aim = c2
        if self2.shape[0]:
            aim = self2[int(np.argmin(np.sum((self2 - pos2) ** 2, axis=1)))]
        delta = aim - pos2
        if float(delta[0] ** 2 + delta[1] ** 2) < 1e-9:
            delta = c2 - pos2
        yaw = math.atan2(float(delta[1]), float(delta[0]))
        off = float(target_dist) if target_dist is not None else float(np.linalg.norm(pos2 - c2))
        return NavigationPose(
            position=(float(p3[0]), float(p3[1]), float(p3[2])),
            yaw_rad=float(yaw),
            clearance_m=float(clearance),
            required_clearance_m=float(required),
            offset_from_target_m=off,
            target_position=(float(c[0]), float(c[1]), float(c[2])),
            navigable=bool(navigable),
            note=note,
            robot_radius_m=float(robot_radius_m),
            clearance_margin_m=float(clearance_margin_m),
        )

    # ---- KD-trees for nearest OTHER-object voxel and nearest TARGET voxel ----
    try:
        from scipy.spatial import cKDTree
    except Exception:
        cKDTree = None

    def _nearest(pointset: np.ndarray, cands: np.ndarray) -> np.ndarray:
        if pointset.shape[0] == 0:
            return np.full(cands.shape[0], np.inf, dtype=np.float64)
        if cKDTree is not None:
            d, _ = cKDTree(pointset).query(cands, k=1)
            return np.asarray(d, dtype=np.float64)
        diff = cands[:, None, :] - pointset[None, :, :]
        return np.sqrt((diff * diff).sum(-1)).min(axis=1)

    # ---- candidate grid around the target centroid ----------------------
    step = max(0.05, float(radial_step_m))
    reach = float(search_radius_m)
    axis = np.arange(-reach, reach + 1e-6, step)
    gx, gy = np.meshgrid(c2[0] + axis, c2[1] + axis)
    cands = np.stack([gx.ravel(), gy.ravel()], axis=1)

    if shared_tree is not None:
        # k-NN against the shared all-voxel tree, drop this target's own voxels.
        k = min(48, shared_owner.shape[0])
        dd, ii = shared_tree.query(cands, k=k)
        dd = np.atleast_2d(dd); ii = np.atleast_2d(ii)
        same = shared_owner[ii] == int(target_index)
        dd_obs = np.where(same, np.inf, dd)
        d_obs = dd_obs.min(axis=1)
    else:
        d_obs = _nearest(obs2, cands)
    d_wall = _bound_clearance(cands)
    d_self = _nearest(self2, cands) if self2.shape[0] else np.linalg.norm(cands - c2[None, :], axis=1)
    body_clear = np.minimum(d_obs, d_wall)

    feasible = (body_clear >= required) & (d_self >= float(min_target_standoff_m))
    if np.any(feasible):
        idx = np.where(feasible)[0]
        # closest feasible pose to the object, biased toward `target_standoff_m`
        cost = np.abs(d_self[idx] - float(target_standoff_m))
        k = idx[int(np.argmin(cost))]
        td = float(d_self[k])
        navigable = td <= float(max_target_dist_m)
        note = (
            f"stand {td:.2f} m from the object (nearest part), "
            f"{float(body_clear[k]) - required:+.2f} m past the {required:.2f} m body+barrier"
        )
        if not navigable:
            note = (
                f"nearest body-safe pose is {td:.2f} m from the object "
                f"(> {max_target_dist_m:.1f} m) — it is boxed in by other objects/walls"
            )
        return _pose(cands[k], float(body_clear[k]), navigable, note, target_dist=td)

    # Nothing satisfies the self-standoff; relax it (allow the object's edge)
    relaxed = body_clear >= required
    if np.any(relaxed):
        idx = np.where(relaxed)[0]
        k = idx[int(np.argmin(d_self[idx]))]
        return _pose(
            cands[k], float(body_clear[k]), False,
            f"only pose clear of other objects/walls is on the target's own footprint "
            f"({float(d_self[k]):.2f} m from it) — approach on foot",
            target_dist=float(d_self[k]),
        )

    # Best effort: maximise body clearance
    k = int(np.argmax(body_clear))
    bounded = workspace_bounds is not None
    return _pose(
        cands[k], float(body_clear[k]), False,
        f"NO body-safe pose near the target"
        f"{' inside the workspace bounds' if bounded else ''}: best clearance "
        f"{float(body_clear[k]):.2f} m < required {required:.2f} m.",
        target_dist=float(d_self[k]),
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

    # Build one KD-tree over every object's voxels; each target reuses it and
    # masks out its own points by owner (rebuilding a ~10^5-point tree per
    # target was the bottleneck for interactive queries).
    shared = None
    up = int(kwargs.get("up_axis", 2)) % 3
    horiz = [a for a in range(3) if a != up]
    if pts.shape[0]:
        try:
            from scipy.spatial import cKDTree

            shared = (cKDTree(np.ascontiguousarray(pts[:, horiz])), owner)
        except Exception:
            shared = None

    out: dict[int, NavigationPose] = {}
    for i in target_indices:
        i = int(i)
        if 0 <= i < means.shape[0]:
            out[i] = compute_navigation_pose(
                i, means=means, voxel_points=pts, voxel_owner=owner,
                _shared_obstacles=shared, **kwargs
            )
    return out
