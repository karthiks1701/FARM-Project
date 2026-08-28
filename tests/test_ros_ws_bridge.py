"""Wire-format + codec tests for the ROS->WebSocket bridge and its client.

Neither ``rclpy`` nor ``websockets`` is imported here — the bridge only pulls
them inside ``_serve`` / the client inside ``_reader`` — so these run anywhere.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bridge = _load("ros_ws_bridge")
client_mod = _load("ros_ws_client")
RosWsClient = client_mod.RosWsClient


class _FakeImageMsg:
    def __init__(self, rgb: np.ndarray, encoding: str):
        h, w = rgb.shape[:2]
        self.height, self.width = h, w
        self.encoding = encoding
        if encoding == "bgr8":
            data = rgb[..., ::-1]
        elif encoding == "mono8":
            data = rgb[..., :1]
        else:
            data = rgb
        self.step = w * data.shape[2]
        self.data = np.ascontiguousarray(data).tobytes()


def test_frame_roundtrip_odom():
    hdr = {
        "type": "odom",
        "stamp": 12.5,
        "frame_id": "odom",
        "child_frame_id": "base",
        "position": [1.0, 2.0, 3.0],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    frame = bridge._frame(hdr)

    c = RosWsClient("ws://unused")
    c._handle(frame)
    stamp, T = c.latest_odom()
    assert stamp == 12.5
    assert T.shape == (4, 4)
    assert np.allclose(T[:3, 3], [1.0, 2.0, 3.0])
    assert np.allclose(T[:3, :3], np.eye(3))


def test_frame_roundtrip_odom_with_world_transform():
    T_map_odom = np.eye(4)
    T_map_odom[:3, 3] = [10.0, 0.0, 0.0]
    hdr = {
        "type": "odom", "stamp": 1.0, "frame_id": "o", "child_frame_id": "b",
        "position": [0.0, 5.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    c = RosWsClient("ws://unused", world_transform=T_map_odom)
    c._handle(bridge._frame(hdr))
    _stamp, T = c.latest_odom()
    assert np.allclose(T[:3, 3], [10.0, 5.0, 0.0])


def test_frame_roundtrip_image():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
    jpeg = bridge._encode_jpeg(rgb, max_side=0, quality=95)
    assert jpeg[:2] == b"\xff\xd8"  # JPEG SOI
    frame = bridge._frame(
        {"type": "image", "stamp": 3.0, "format": "jpeg", "width": 64, "height": 48}, jpeg
    )

    c = RosWsClient("ws://unused")
    c._handle(frame)
    stamp, img = c.latest_image()
    assert stamp == 3.0
    assert img.shape == (48, 64, 3)
    assert img.dtype == np.uint8


def test_image_want_image_false_drops_frames():
    rgb = np.zeros((8, 8, 3), np.uint8)
    frame = bridge._frame(
        {"type": "image", "stamp": 1.0, "width": 8, "height": 8},
        bridge._encode_jpeg(rgb, 0, 80),
    )
    c = RosWsClient("ws://unused", want_image=False)
    c._handle(frame)
    assert c.latest_image() == (None, None)


@pytest.mark.parametrize("encoding", ["rgb8", "bgr8", "mono8"])
def test_image_msg_to_rgb_encodings(encoding):
    rgb = np.dstack([
        np.full((6, 5), 200, np.uint8),
        np.full((6, 5), 100, np.uint8),
        np.full((6, 5), 50, np.uint8),
    ])
    out = bridge._image_msg_to_rgb(_FakeImageMsg(rgb, encoding))
    assert out.shape == (6, 5, 3)
    assert out.flags["C_CONTIGUOUS"]
    if encoding in ("rgb8", "bgr8"):
        assert np.array_equal(out, rgb)
    else:  # mono8 -> gray replicated across channels
        assert np.array_equal(out[..., 0], out[..., 1])


def test_handle_ignores_short_or_garbage_frames():
    c = RosWsClient("ws://unused")
    c._handle(b"")
    c._handle(b"\x00\x00")
    c._handle(b"\x00\x00\x00\x99short")  # header longer than message
    assert c.latest_odom() == (None, None)
    assert c.latest_image() == (None, None)
