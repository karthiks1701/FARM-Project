#!/usr/bin/env python3
"""Query a saved scene_state.pt with a free-text description.

Loads a scene graph produced by ``python -m scene_graph.offline.run`` (or
``scripts/run_pipeline.py``), builds a :class:`SceneGraphRetriever` over it,
and prints ranked object clusters for a text query.

Run inside the docker container, with an embedding server reachable
(``./run.sh vllm`` starts one)::

    ./scripts/in_docker.sh python scripts/query_scene_graph.py \\
        --pt /data/out/scene0000_00.pt --query "a red backpack"

Set ``QWEN3_VL_EMBED_ENABLED=0`` to skip the VL-embedding path if only the
text embed server is up.

For each relational match the CLI also prints a **collision-aware navigation
pose**: the object centroid is inside the object and unsafe as a robot goal, so
``scene_graph.retrieval.navigation_pose`` derives a nearby standoff pose clear of
every other object's voxel evidence by at least ``--robot-radius-m`` +
``--clearance-m`` (Spot ~0.6 m footprint + a few-cm safety barrier). Matches with
no body-safe pose within ``--nav-search-radius-m`` are flagged ``[UNSAFE]``
rather than silently returned.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt", required=True, help="Path to a saved scene_state.pt file.")
    parser.add_argument("--query", required=True, help="Free-text query, e.g. 'a red backpack near the door'.")
    parser.add_argument("--top-k", type=int, default=5, help="How many ranked results to print.")
    parser.add_argument("--spatial-method", default="joint_v1",
                        help="Relational engine (joint_v1 default; unified_soft_w50 = the paper's locked protocol).")
    parser.add_argument("--embedding-only", action="store_true",
                        help="Skip the LLM parse + relational scoring; embedding cluster retrieval only.")
    parser.add_argument("--no-nav-pose", action="store_true",
                        help="Don't compute a collision-aware navigation pose for each match (relational path only).")
    parser.add_argument("--clearance-m", type=float, default=0.10,
                        help="Safety barrier (metres) kept between the robot footprint and any other object's voxels.")
    parser.add_argument("--robot-radius-m", type=float, default=0.6,
                        help="Robot footprint half-width (metres); the nav pose keeps robot_radius + clearance from obstacles.")
    parser.add_argument("--nav-search-radius-m", type=float, default=2.5,
                        help="How far out from the object to search for a body-safe standoff pose.")
    parser.add_argument("--nav-up-axis", type=int, default=2, choices=(0, 1, 2),
                        help="World vertical axis (Spot/GraphNav seed frame is Z-up = 2).")
    args = parser.parse_args()

    pt_path = Path(args.pt)
    if not pt_path.exists():
        print(f"scene_state.pt not found at {pt_path}")
        return 2

    from scene_graph.llm_utils import EmbedInterface, LLMInterface
    from scene_graph.scene_state_io import load_scene_state

    embedder = EmbedInterface(verbose=False)

    if not args.embedding_only:
        # Relational path (same pipeline as the viser Query panel): LLM parse
        # -> execute_spatial_query. Falls back to embedding clusters when the
        # parser LLM is unreachable.
        try:
            import torch

            from scene_graph.retrieval.spatial_reasoning import execute_spatial_query, parse_query

            payload = torch.load(pt_path, map_location="cpu", weights_only=False)
            feature_dim = payload.get("feature_dim") if isinstance(payload, dict) else None
            state = (
                load_scene_state(pt_path, feature_dim=int(feature_dim), device="cpu")
                if feature_dim is not None
                else payload.get("state", payload)
            )
            captions = state.get("object_caption") or []
            means = state.get("means")
            llm = LLMInterface(verbose=False)
            t0 = time.time()
            query_graph = parse_query(args.query, llm)
            scored = execute_spatial_query(
                query_graph, state, llm, embedder,
                use_vlm=False, pre_filter_k=-1, max_output_candidates=max(args.top_k, 20),
                raw_query=args.query, retrieval_mode="multi", candidate_pool_mode="active",
                spatial_method=str(args.spatial_method), verbose=False,
            )
            dt = time.time() - t0
            preds = ", ".join(f"{p.name}({', '.join(map(str, p.args))})" for p in query_graph.predicates)
            print(f"\nquery={args.query!r} [{args.spatial_method}] -> parsed [{preds}] in {dt:.2f}s\n")

            nav_poses = {}
            if not args.no_nav_pose:
                try:
                    from scene_graph.retrieval.navigation_pose import navigation_poses_for_scene

                    nav_poses = navigation_poses_for_scene(
                        state,
                        [c.object_index for c in scored[: args.top_k]],
                        clearance_margin_m=float(args.clearance_m),
                        robot_radius_m=float(args.robot_radius_m),
                        search_radius_m=float(args.nav_search_radius_m),
                        up_axis=int(args.nav_up_axis),
                    )
                except Exception as exc:  # noqa: BLE001 - nav pose is advisory
                    print(f"(navigation-pose computation skipped: {exc})\n")

            for rank, cand in enumerate(scored[: args.top_k], start=1):
                cap = str(captions[cand.object_index] if cand.object_index < len(captions) else "") or "(no caption)"
                cap = (cap[:70] + "…") if len(cap) > 70 else cap
                anchors = ", ".join(str(a) for a in (cand.matched_anchors or {}).values())
                extra = f" anchors=[{anchors}]" if anchors else ""
                pos_str = "pos=?"
                if means is not None and 0 <= cand.object_index < len(means):
                    x, y, z = (float(v) for v in means[cand.object_index])
                    pos_str = f"pos=({x:.2f}, {y:.2f}, {z:.2f})"
                print(f"  #{rank} object_id={cand.object_id} score={cand.composite_score:.3f} {pos_str}{extra} {cap!r}")
                nav = nav_poses.get(cand.object_index)
                if nav is not None:
                    nx, ny, _nz = nav.position
                    print(f"       {nav.summary()}  — {nav.note}")
                    # Machine-readable seed-frame goal for a downstream planner:
                    #   NAVGOAL <object_id> <x> <y> <yaw_rad> <navigable> <clearance_m>
                    print(
                        f"       NAVGOAL {cand.object_id} {nx:.4f} {ny:.4f} "
                        f"{nav.yaw_rad:.5f} {int(bool(nav.navigable))} {nav.clearance_m:.3f}"
                    )
            return 0
        except Exception as exc:  # noqa: BLE001 - degrade to embedding retrieval
            print(f"Relational path unavailable ({exc}); falling back to embedding retrieval.")

    from scene_graph.retrieval.scene_graph_retriever import SceneGraphRetriever

    retriever = SceneGraphRetriever.from_scene_state(pt_path, embedder=embedder, verbose=False)
    print(f"Loaded {len(retriever._processor.object_ids)} objects from {pt_path}")

    t0 = time.time()
    result = retriever.retrieve(args.query)
    dt = time.time() - t0

    clusters = result.get("clusters", []) or []
    print(f"\nquery={args.query!r} -> {len(clusters)} clusters in {dt:.2f}s\n")
    for i, cluster in enumerate(clusters[: args.top_k]):
        score = cluster.get("cluster_score", 0.0)
        candidates = cluster.get("candidate_objects", []) or []
        print(f"  #{i + 1} score={score:.3f} ({len(candidates)} candidates)")
        for cand in candidates[:3]:
            caption = str(cand.get("caption") or cand.get("object_caption") or "")
            caption = (caption[:70] + "…") if len(caption) > 70 else caption
            pos = cand.get("position") or [0.0, 0.0, 0.0]
            pos_str = f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
            print(f"      object_id={cand.get('object_id')} score={cand.get('final_retrieval_score', 0.0):.3f} {pos_str} {caption!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
