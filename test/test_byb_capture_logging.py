from __future__ import annotations

import json

import daq.backyard_brains as byb


def test_capture_session_writes_stream_and_events(tmp_path) -> None:
    capture = byb._BYBCaptureSession(
        meta={"device": "/dev/tty.mfi", "transport": "serial"},
        base_dir=tmp_path,
        max_stream_bytes=4,
    )

    capture.record_write("max:;")
    capture.record_read(b"\x80\x01ABCD", timeout_ms=25)
    capture.record_message("probe_text", "MSF:20000MNC:2;")
    capture.close()

    stream_path = capture.path / "stream.bin"
    events_path = capture.path / "events.jsonl"

    assert stream_path.read_bytes() == b"\x80\x01AB"

    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == [
        "session_start",
        "write_command",
        "read_chunk",
        "protocol_message",
        "session_end",
    ]
    assert events[2]["stored"] == 4
    assert events[2]["dropped"] == 2
    assert events[3]["message"] == "MSF:20000MNC:2;"


def test_text_protocol_filter_rejects_binary_garbage() -> None:
    assert byb._looks_like_text_protocol_message("MSF:20000;") is True
    assert byb._looks_like_text_protocol_message("BRD:0;") is True
    assert byb._looks_like_text_protocol_message("FWV:1.00;") is True
    assert byb._looks_like_text_protocol_message("MSF:20000MNC:2;") is True

    assert byb._looks_like_text_protocol_message("") is False
    assert byb._looks_like_text_protocol_message("not-a-message") is False
    assert byb._looks_like_text_protocol_message(";\x00\x01garbage") is False
    assert byb._looks_like_text_protocol_message("M\x88F:20000;") is False
