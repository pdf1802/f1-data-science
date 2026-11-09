"""
Track view — the race drawn on the real circuit, live from the stream.

    python -m integrations.f1_race_replay.track_view                 # its own window on localhost:9999
    (or the TRACK MAP button in the Race Engineer window: same stream, no second connection)

Works with both streams:
  * the stand-in stream (replay_stream.py), including TAKE CONTROL: its cars are placed
    by their fraction of the lap (`rel_dist`) on the circuit outline it sends as
    `track_geometry` (tools/circuit_outlines.json, from GPS laps in the FastF1 cache), and
    animated smoothly between payloads (the stream only sends ~4 per second);
  * the real f1-race-replay: its frames carry real GPS positions (x, y) and its own
    `track_geometry` — drawn as they come.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget

BG, TRACK, TRACK_EDGE, TEXT, DIM = "#0f1012", "#3a3d44", "#55595f", "#ececec", "#8b8f97"
STATUS = {"4": ("SAFETY CAR", "#f5a524"), "5": ("RED FLAG", "#e10600"), "6": ("VIRTUAL SAFETY CAR", "#f5a524"),
          "7": ("VSC ENDING", "#f5a524")}


class TrackMap(QWidget):
    """Draws the latest payload: outline, cars, lap and mode. Feed it with set_payload()."""

    def __init__(self, driver: str | None = None, parent=None):
        super().__init__(parent)
        self.setMinimumSize(520, 420)
        self.driver = (driver or "").upper() or None
        self.payload: dict | None = None
        self.geometry_xy: np.ndarray | None = None          # outline, already rotated
        self.rotation = 0.0
        self._cum: np.ndarray | None = None                  # cumulative arc length of the outline
        self._frac: dict[str, tuple[float, float]] = {}      # driver → (previous, current) lap fraction
        self._t: tuple[float, float] = (0.0, 0.0)            # wall time of those two payloads
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(33)                                 # ~30 fps animation

    # ── data ────────────────────────────────────────────────────────────────
    def set_payload(self, p: dict):
        geo = p.get("track_geometry")
        if geo and geo.get("x"):
            self._set_geometry(geo)
        elif self.geometry_xy is None and p.get("position_source") == "lap-fraction":
            # stand-in stream on a circuit with no GPS outline: it places cars on a circle — draw that circle
            R = float(p.get("circuit_length_m") or 5000) / (2 * math.pi)
            a = np.linspace(0, 2 * math.pi, 400, endpoint=False)
            self._set_geometry({"x": (R * np.cos(a)).tolist(), "y": (R * np.sin(a)).tolist(), "rotation_deg": 0.0})
        now = time.monotonic()
        drivers = (p.get("frame") or {}).get("drivers") or {}
        new = {}
        for d, fd in drivers.items():
            cur = (fd.get("lap") or 1) - 1 + (fd.get("rel_dist") or 0.0)
            prev = self._frac.get(d, (cur, cur))[1]
            if cur < prev - 0.5:                              # rewind / restart: jump
                prev = cur
            new[d] = (prev, cur)
        self._frac = new
        self._t = (self._t[1], now)
        self.payload = p

    def _set_geometry(self, geo: dict):
        rot = float(geo.get("rotation_deg") or 0.0)
        if self.geometry_xy is not None and rot == self.rotation and len(geo["x"]) == len(self.geometry_xy):
            return
        x, y = np.asarray(geo["x"], float), np.asarray(geo["y"], float)
        self.rotation = rot
        self.geometry_xy = self._rotate(x, y)
        closed = np.vstack([self.geometry_xy, self.geometry_xy[:1]])
        self._cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(closed, axis=0).T))])

    def _rotate(self, x, y):
        r = math.radians(self.rotation)
        return np.column_stack([x * math.cos(r) - y * math.sin(r), x * math.sin(r) + y * math.cos(r)])

    def _on_outline(self, rel: float) -> tuple[float, float]:
        """Point at fraction `rel` of the outline's length (the outline starts at the timing line)."""
        closed = np.vstack([self.geometry_xy, self.geometry_xy[:1]])
        s = (rel % 1.0) * self._cum[-1]
        return float(np.interp(s, self._cum, closed[:, 0])), float(np.interp(s, self._cum, closed[:, 1]))

    def _car_xy(self, d: str, fd: dict, lap_fraction: bool) -> tuple[float, float] | None:
        if lap_fraction and self.geometry_xy is not None:
            prev, cur = self._frac.get(d, (None, None))
            if cur is None:
                return None
            t0, t1 = self._t
            a = min(max((time.monotonic() - t1) / max(t1 - t0, 1e-3), 0.0), 1.0)   # one payload behind
            return self._on_outline(prev + a * (cur - prev))
        if fd.get("x") is None or fd.get("y") is None:
            return None
        p = self._rotate(np.array([float(fd["x"])]), np.array([float(fd["y"])]))[0]
        return float(p[0]), float(p[1])

    # ── drawing ─────────────────────────────────────────────────────────────
    def paintEvent(self, _event):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        qp.fillRect(self.rect(), QColor(BG))
        p = self.payload
        if not p or self.geometry_xy is None:
            qp.setPen(QColor(DIM))
            qp.drawText(self.rect(), Qt.AlignCenter, "Waiting for the stream (and the track geometry)…")
            return
        xy = self.geometry_xy
        top, margin = 54, 26
        area = QRectF(margin, top, self.width() - 2 * margin, self.height() - top - margin)
        (x0, y0), (x1, y1) = xy.min(axis=0), xy.max(axis=0)
        scale = min(area.width() / max(x1 - x0, 1), area.height() / max(y1 - y0, 1))
        ox = area.left() + (area.width() - (x1 - x0) * scale) / 2
        oy = area.top() + (area.height() - (y1 - y0) * scale) / 2

        def to_screen(x, y):
            return QPointF(ox + (x - x0) * scale, oy + (y1 - y) * scale)   # screen y grows downwards

        path = QPainterPath(to_screen(*xy[0]))
        for x, y in xy[1:]:
            path.lineTo(to_screen(x, y))
        path.closeSubpath()
        qp.setPen(QPen(QColor(TRACK_EDGE), 14, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        qp.drawPath(path)
        qp.setPen(QPen(QColor(TRACK), 10, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        qp.drawPath(path)
        # timing line
        a, b = to_screen(*xy[0]), to_screen(*xy[1])
        n = QPointF(-(b - a).y(), (b - a).x())
        n *= 9 / max(math.hypot(n.x(), n.y()), 1e-6)
        qp.setPen(QPen(QColor("white"), 3))
        qp.drawLine(a - n, a + n)

        lap_fraction = p.get("position_source") == "lap-fraction"
        drivers = (p.get("frame") or {}).get("drivers") or {}
        colours = p.get("driver_colors") or {}
        order = sorted(drivers, key=lambda d: -(drivers[d].get("position") or 99))   # leader drawn last
        if self.driver in order:
            order.remove(self.driver)
            order.append(self.driver)
        qp.setFont(QFont("DejaVu Sans", 8, QFont.Bold))
        for d in order:
            fd = drivers[d]
            pos = self._car_xy(d, fd, lap_fraction)
            if pos is None:
                continue
            c = to_screen(*pos)
            me = d == self.driver
            r = 9 if me else 6
            col = QColor(colours.get(d, "#cccccc"))
            if fd.get("in_pit"):
                col.setAlpha(110)
            qp.setPen(QPen(QColor("white") if me else QColor(BG), 2.5 if me else 1.2))
            qp.setBrush(col)
            qp.drawEllipse(c, r, r)
            qp.setPen(QColor(TEXT if me or (fd.get("position") or 99) <= 10 else DIM))
            qp.drawText(c + QPointF(r + 2, -r + 2), d)

        # header
        sd = p.get("session_data") or {}
        info = p.get("session_info") or {}
        tc = p.get("take_control") or {}
        qp.setPen(QColor(TEXT))
        qp.setFont(QFont("DejaVu Sans", 12, QFont.Bold))
        title = f"{info.get('year', '')} {info.get('event_name', '')}".strip() or "LIVE"
        qp.drawText(QPointF(16, 24), f"{title.upper()}   ·   LAP {sd.get('lap', '?')}/{sd.get('total_laps', '?')}")
        qp.setFont(QFont("DejaVu Sans", 9, QFont.Bold))
        if tc.get("active"):
            mode, mcol = f"ENGINE IN CONTROL SINCE LAP {tc.get('from_lap')}", "#f5a524"
        elif tc.get("from_lap"):
            mode, mcol = f"REAL RACE — ENGINE TAKES OVER AFTER LAP {tc.get('from_lap')}", DIM
        else:
            mode, mcol = ("REAL RACE (STAND-IN STREAM)" if lap_fraction else "F1-RACE-REPLAY · REAL GPS"), DIM
        status = STATUS.get(str(p.get("track_status") or "1"))
        qp.setPen(QColor(mcol))
        qp.drawText(QPointF(16, 42), mode + ("   ·   positions: lap fraction on the GPS outline" if lap_fraction else ""))
        if status:
            qp.setPen(QColor(status[1]))
            qp.setFont(QFont("DejaVu Sans", 12, QFont.Bold))
            qp.drawText(QPointF(self.width() - 220, 24), status[0])
        qp.end()


class TrackViewWindow(QMainWindow):
    """Standalone window with its own connection to the stream."""

    def __init__(self, host="localhost", port=None, driver=None):
        super().__init__()
        from integrations.f1_race_replay.race_engineer_window import StreamClient
        from integrations.f1_race_replay.replay_stream import PORT
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("F1 Race Replay — Track view")
        self.resize(900, 640)
        self.map = TrackMap(driver)
        self.setCentralWidget(self.map)
        self.client = StreamClient(host, port or PORT)
        self.client.data_received.connect(self.map.set_payload)
        self.client.start()

    def closeEvent(self, event):
        self.client.stop()
        event.accept()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--driver", default=None, help="highlight this driver, e.g. ALO")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    w = TrackViewWindow(args.host, args.port, args.driver)
    app.aboutToQuit.connect(w.close)
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
