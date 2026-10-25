from __future__ import annotations

from collections import deque
import sys
import threading
from pathlib import Path


READER_ROOT = Path(__file__).resolve().parents[2] / "dependency" / "oculus_reader"
if str(READER_ROOT) not in sys.path:
    sys.path.insert(0, str(READER_ROOT))

from oculus_reader.reader import OculusReader  # noqa: E402


def test_reader_buffers_every_parsed_logcat_frame_with_arrival_time():
    reader = object.__new__(OculusReader)
    reader.running = True
    reader.tag = "wE9ryARX"
    reader.print_FPS = False
    reader._lock = threading.Lock()
    reader.last_transforms = {}
    reader.last_buttons = {}
    reader.last_tracking = {}
    reader._sample_sequence = 0
    reader._samples = deque(maxlen=256)

    matrix = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"

    class File:
        def readline(self):
            reader.running = False
            return f"I/test wE9ryARX: r:{matrix}&R,RG"

        def close(self):
            pass

    class Socket:
        def makefile(self):
            return File()

    class Connection:
        socket = Socket()

        def close(self):
            pass

    reader.read_logcat_by_line(Connection())

    latest, samples = reader.get_samples_since(0)
    assert latest == 1
    assert len(samples) == 1
    sequence, received_at, transforms, buttons, tracking = samples[0]
    assert sequence == 1
    assert received_at > 0.0
    assert set(transforms) == {"r"}
    assert buttons["RG"] is True
    assert tracking == {}


def test_reader_parses_v2_world_tracking_twist_and_metadata():
    matrix = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"
    packet = (
        f"r:{matrix}&R,RG&v2,tracking_world&"
        "r:4 12.5 0.01 3 1 1 2 3 4 5 6"
    )

    transforms, buttons, tracking = OculusReader.process_packet(packet)

    assert set(transforms) == {"r"}
    assert buttons["RG"] is True
    assert tracking["r"]["frame"] == "tracking_world"
    assert tracking["_protocol_version"] == 2
    assert tracking["_frame"] == "tracking_world"
    assert tracking["r"]["device_type"] == 4
    assert tracking["r"]["time_in_seconds"] == 12.5
    assert tracking["r"]["prediction_in_seconds"] == 0.01
    assert tracking["r"]["status"] == 3
    assert tracking["r"]["confidence"] == 1
    assert tracking["r"]["linear_velocity"].tolist() == [1.0, 2.0, 3.0]
    assert tracking["r"]["angular_velocity"].tolist() == [4.0, 5.0, 6.0]


def test_legacy_process_data_ignores_v2_extension_for_compatibility():
    matrix = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"
    packet = (
        f"l:{matrix}&L,LG&v2,tracking_world&"
        "l:4 12.5 0.01 3 1 1 2 3 4 5 6"
    )

    transforms, buttons = OculusReader.process_data(packet)

    assert set(transforms) == {"l"}
    assert buttons["LG"] is True


def test_malformed_v2_tracking_record_preserves_protocol_marker():
    matrix = "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"
    packet = f"r:{matrix}&R,RG&v2,tracking_world&r:not-numbers"

    _transforms, _buttons, tracking = OculusReader.process_packet(packet)

    assert tracking == {
        "_protocol_version": 2,
        "_frame": "tracking_world",
    }
