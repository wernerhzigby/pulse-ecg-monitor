import os
import time
from dataclasses import dataclass, field
from collections import deque, defaultdict

CARDIAC_EVENTS = {
    "Bradycardia",
    "Tachycardia",
    "Ventricular Tachycardia",
    "Asystole / Flatline",
    "Irregular Rhythm",
    "Sinus Node Dysfunction",
    "First-Degree AV Block (possible)",
    "Bundle Branch Block (possible)",
    "Long QT (possible)",
    "Short QT (possible)",
    "Early Repolarization / ST Elevation (possible)",
    "Myocarditis (possible)",
}


@dataclass
class ECGConfig:
    sample_rate: int = int(os.getenv("ECG_SAMPLE_RATE", "250"))
    r_threshold: int = int(os.getenv("ECG_R_THRESHOLD", "15000"))
    brady_bpm: int = int(os.getenv("ECG_BRADY_BPM", "50"))
    tachy_bpm: int = int(os.getenv("ECG_TACHY_BPM", "100"))
    vtach_bpm: int = int(os.getenv("ECG_VTACH_BPM", "150"))
    asystole_sec: float = float(os.getenv("ECG_ASYSTOLE_SEC", "3.5"))
    buffer_sec: int = int(os.getenv("ECG_BUFFER_SEC", "120"))
    bpm_maxlen: int = int(os.getenv("ECG_BPM_MAXLEN", "1200"))
    rr_maxlen: int = int(os.getenv("ECG_RR_MAXLEN", "60"))
    qrs_maxlen: int = int(os.getenv("ECG_QRS_MAXLEN", "30"))
    qt_maxlen: int = int(os.getenv("ECG_QT_MAXLEN", "30"))

    @property
    def ecg_maxlen(self) -> int:
        return max(1000, self.sample_rate * self.buffer_sec)


@dataclass
class ECGState:
    config: ECGConfig
    ecg_data: deque = field(init=False)
    timestamps: deque = field(init=False)
    bpm_history: deque = field(init=False)
    bpm_timestamps: deque = field(init=False)
    rr_intervals: deque = field(init=False)
    qrs_widths: deque = field(init=False)
    qt_intervals: deque = field(init=False)
    event_state: dict = field(default_factory=dict)
    event_counts: defaultdict = field(default_factory=lambda: defaultdict(int))
    event_timeline: deque = field(init=False)
    current_bpm: int = 0
    last_peak_time: float | None = None
    last_signal_time: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.ecg_data = deque(maxlen=self.config.ecg_maxlen)
        self.timestamps = deque(maxlen=self.config.ecg_maxlen)
        self.bpm_history = deque(maxlen=self.config.bpm_maxlen)
        self.bpm_timestamps = deque(maxlen=self.config.bpm_maxlen)
        self.rr_intervals = deque(maxlen=self.config.rr_maxlen)
        self.qrs_widths = deque(maxlen=self.config.qrs_maxlen)
        self.qt_intervals = deque(maxlen=self.config.qt_maxlen)
        self.event_timeline = deque(maxlen=self.config.ecg_maxlen)

    def reset(self) -> None:
        self.ecg_data.clear()
        self.timestamps.clear()
        self.bpm_history.clear()
        self.bpm_timestamps.clear()
        self.rr_intervals.clear()
        self.qrs_widths.clear()
        self.qt_intervals.clear()
        self.event_state.clear()
        self.event_counts.clear()
        self.event_timeline.clear()
        self.current_bpm = 0
        self.last_peak_time = None
        self.last_signal_time = time.time()

    def set_event(self, name: str, condition: bool) -> None:
        if condition:
            self.event_state[name] = True
            self.event_counts[name] += 1
        else:
            self.event_state.pop(name, None)

    def active_cardiac_flags(self) -> list[str]:
        return [e for e in self.event_state if e in CARDIAC_EVENTS]

    def add_sample(self, value: int, t: float) -> None:
        self.ecg_data.append(value)
        self.timestamps.append(t)

        if value > self.config.r_threshold:
            if self.last_peak_time:
                rr = t - self.last_peak_time
                if rr > 0.25:
                    self.rr_intervals.append(rr)
                    bpm = 60 / rr
                    self.current_bpm = int(bpm)
                    self.bpm_history.append(self.current_bpm)
                    self.bpm_timestamps.append(t)

                    self.qt_intervals.append(rr * 0.45)
                    self.qrs_widths.append(0.08 + abs(value - self.config.r_threshold) / 100000)

            self.last_peak_time = t
            self.last_signal_time = t

        self.detect_events(value, t)
        self.event_timeline.append(",".join(self.active_cardiac_flags()))

    def detect_events(self, value: int, now: float) -> None:
        self.set_event("Bradycardia", self.current_bpm and self.current_bpm < self.config.brady_bpm)
        self.set_event("Tachycardia", self.current_bpm and self.current_bpm > self.config.tachy_bpm)
        self.set_event(
            "Ventricular Tachycardia",
            self.current_bpm and self.current_bpm > self.config.vtach_bpm,
        )

        self.set_event("Asystole / Flatline", now - self.last_signal_time > self.config.asystole_sec)

        if len(self.rr_intervals) > 6:
            mean_rr = sum(self.rr_intervals) / len(self.rr_intervals)
            variance = sum((r - mean_rr) ** 2 for r in self.rr_intervals) / len(self.rr_intervals)

            self.set_event("Irregular Rhythm", variance > 0.02)
            self.set_event("Sinus Node Dysfunction", variance > 0.03 and mean_rr > 1.2)
            self.set_event("First-Degree AV Block (possible)", mean_rr > 1.0 and variance < 0.005)

        if len(self.qrs_widths) > 5:
            self.set_event(
                "Bundle Branch Block (possible)",
                sum(self.qrs_widths) / len(self.qrs_widths) > 0.14,
            )

        if len(self.qt_intervals) > 5:
            avg_qt = sum(self.qt_intervals) / len(self.qt_intervals)
            self.set_event("Long QT (possible)", avg_qt > 0.48)
            self.set_event("Short QT (possible)", avg_qt < 0.32)

        self.set_event(
            "Early Repolarization / ST Elevation (possible)",
            value > self.config.r_threshold * 1.25 and self.current_bpm < 100,
        )

        myocarditis_score = 0
        if "Tachycardia" in self.event_state:
            myocarditis_score += 1
        if "Irregular Rhythm" in self.event_state:
            myocarditis_score += 1
        if "Early Repolarization / ST Elevation (possible)" in self.event_state:
            myocarditis_score += 1

        self.set_event("Myocarditis (possible)", myocarditis_score >= 2)
