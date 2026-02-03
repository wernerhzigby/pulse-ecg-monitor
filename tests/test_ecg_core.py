import time

from ecg_core import ECGConfig, ECGState


def make_state():
    config = ECGConfig(sample_rate=250, r_threshold=15000)
    return ECGState(config)


def test_bradycardia_detection():
    state = make_state()
    state.current_bpm = 40
    state.detect_events(value=0, now=time.time())
    assert "Bradycardia" in state.event_state


def test_tachycardia_detection():
    state = make_state()
    state.current_bpm = 120
    state.detect_events(value=0, now=time.time())
    assert "Tachycardia" in state.event_state


def test_asystole_detection():
    state = make_state()
    state.last_signal_time = 0.0
    state.detect_events(value=0, now=10.0)
    assert "Asystole / Flatline" in state.event_state


def test_repolarization_detection():
    state = make_state()
    state.current_bpm = 80
    state.detect_events(value=int(state.config.r_threshold * 1.3), now=time.time())
    assert "Early Repolarization / ST Elevation (possible)" in state.event_state
