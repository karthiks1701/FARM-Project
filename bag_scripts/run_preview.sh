#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash
source /home/scene_graph/.venv/bin/activate
source /tmp/colcon_ws/install/setup.bash
python -m scene_graph.offline.run \
    --source npz \
    --npz-dir /data/out/ria_rgb_d_mistlab_1_npz \
    --camera oak_rgb \
    --save-path /data/out/ria_rgb_d_mistlab_1_preview25.pt \
    --covisibility \
    --viser \
    --end 5000
