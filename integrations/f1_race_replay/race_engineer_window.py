"""
Race Engineer — a pit-wall window for f1-race-replay's telemetry stream.

    python -m integrations.f1_race_replay.race_engineer_window --driver ALO
    python -m integrations.f1_race_replay.race_engineer_window --driver ALO --circuit "Bahrain Grand Prix"

Connects to localhost:9999 — f1-race-replay started with `--telemetry`, or the
stand-in stream (replay_stream.py) — and, each time the driver you follow
crosses the line, shows the strategy call of tools/race_engineer.py:

    headline        BOX NOW → HARD — +1.4 places expected vs staying out
    options         P(win / podium / points), expected position, gain vs staying out ± 95 % CI
    timing tower    the race at the end of that lap (~ = line time estimated: not crossed yet)
    undercut alert  when the car ahead is within 3 s
    take control    box / stay-out buttons when the stream runs in take-control mode

Threads: the socket is read by StreamClient (a QThread, same protocol as
f1-race-replay's TelemetryStreamClient); the Monte Carlo runs in EngineerWorker
(another QThread) on the most recent payload only, so the UI never blocks and
never falls behind a fast replay.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
import traceback

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QFrame, QGridLayout,
                               QHBoxLayout, QHeaderView, QLabel, QListWidget, QMainWindow,
                               QPushButton, QSizePolicy, QSpinBox, QStatusBar, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from integrations.f1_race_replay.replay_stream import CIRCUIT_LENGTH_M, PORT
from tools.race_engineer import RaceEngineer, completed_laps, tyre_name

# ── look ─────────────────────────────────────────────────────────────────────
BG, PANEL, PANEL_2, LINE = "#0f1012", "#17181b", "#1e2024", "#2a2d33"
TEXT, DIM, ACCENT = "#ececec", "#8b8f97", "#e10600"
GOOD, WARN, INFO = "#2fbf71", "#f5a524", "#4ea1ff"
TYRE_COLOUR = {"SOFT": "#e8002d", "MEDIUM": "#ffd12e", "HARD": "#f0f0ec", "INTERMEDIATE": "#39b54a", "WET": "#2f6bff"}
SANS = '"Segoe UI", "Inter", "DejaVu Sans", sans-serif'
MONO = '"Consolas", "DejaVu Sans Mono", monospace'

STYLE = f"""
QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; font-family: {SANS}; font-size: 13px; }}
QFrame#panel {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 8px; }}
QFrame#panel QLabel, QFrame#panel QWidget#bare {{ background: transparent; }}
QLabel#title {{ font-size: 15px; font-weight: 700; letter-spacing: 2px; }}
QLabel#caption {{ color: {DIM}; font-size: 11px; font-weight: 600; letter-spacing: 1px; }}
QLabel#headline {{ font-size: 22px; font-weight: 800; padding: 2px 0; }}
QLabel#sub {{ color: {DIM}; }}
QLabel#note {{ color: {DIM}; font-size: 11px; }}
QComboBox, QSpinBox {{ background: {PANEL_2}; border: 1px solid {LINE}; border-radius: 4px; padding: 3px 6px; min-height: 22px; }}
QTableWidget {{ background: {PANEL}; gridline-color: {LINE}; border: none; font-family: {MONO}; font-size: 12px;
               selection-background-color: {PANEL_2}; selection-color: {TEXT}; }}
QHeaderView::section {{ background: {PANEL}; color: {DIM}; border: none; border-bottom: 1px solid {LINE};
                       padding: 4px; font-family: {SANS}; font-size: 11px; font-weight: 600; }}
QPushButton {{ background: {PANEL_2}; border: 1px solid {LINE}; border-radius: 6px; padding: 8px 12px; font-weight: 700; }}
QPushButton:hover {{ border-color: {DIM}; }}
QPushButton:disabled {{ color: #555; }}
QListWidget {{ background: {PANEL}; border: none; font-family: {MONO}; font-size: 11px; color: {DIM}; }}
QStatusBar {{ color: {DIM}; }}
"""


def guess_circuit(payload: dict) -> tuple[str | None, str]:
    """
    Circuit for the strategy models, and where it came from. The real payload has
    no event name (proposed upstream as `session_info`), so fall back to the
    circuit length it does send — unique for most tracks, not for Bahrain/Miami.
    """
    name = (payload.get("session_info") or {}).get("event_name")
    if name:
        return name, "from the stream"
    length = payload.get("circuit_length_m")
    if length:
        close = [c for c, m in CIRCUIT_LENGTH_M.items() if abs(m - float(length)) / m < 0.01]
        if len(close) == 1:
            return f"{close[0]} Grand Prix", "guessed from the track length"
        if close:
            return None, "ambiguous track length (" + " / ".join(close) + ") — pick the circuit"
    return None, "unknown circuit — pick it (generic profile meanwhile)"


# ── threads ──────────────────────────────────────────────────────────────────
class StreamClient(QThread):
    """f1-race-replay's TelemetryStreamClient protocol (JSON lines over TCP), plus send()."""

    data_received = Signal(object)
    connection_status = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, host: str = "localhost", port: int = PORT):
        super().__init__()
        self.host, self.port = host, port
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            try:
                self.connection_status.emit("Connecting…")
                sock = socket.create_connection((self.host, self.port), timeout=5)
                sock.settimeout(1.0)
                with self._lock:
                    self._sock = sock
                self.connection_status.emit("Connected")
                self._read(sock)
            except OSError as e:
                if self._running:
                    self.error_occurred.emit(f"{e} — is the replay running with --telemetry?")
            finally:
                with self._lock:
                    if self._sock:
                        self._sock.close()
                    self._sock = None
            if self._running:
                self.connection_status.emit("Disconnected")
                self.msleep(1500)

    def _read(self, sock):
        buf = b""
        while self._running:
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    try:
                        self.data_received.emit(json.loads(line))
                    except ValueError as e:
                        self.error_occurred.emit(f"Bad JSON: {e}")

    def send(self, message: dict) -> bool:
        """Send one JSON line to the server (take-control commands)."""
        with self._lock:
            if not self._sock:
                return False
            try:
                self._sock.sendall((json.dumps(message) + "\n").encode())
                return True
            except OSError:
                return False

    def stop(self):
        self._running = False
        with self._lock:
            if self._sock:
                try:
                    self._sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        self.wait(3000)


class EngineerWorker(QThread):
    """Runs RaceEngineer.update on the LATEST payload only — older ones are dropped, never queued."""

    call_ready = Signal(object, float)        # call, seconds it took
    busy = Signal(bool)
    failed = Signal(str)

    def __init__(self, n_sims: int = 300):
        super().__init__()
        self.n_sims = n_sims
        self._cond = threading.Condition()
        self._payload: dict | None = None
        self._config: tuple | None = None
        self._last: dict | None = None
        self._running = True

    def submit(self, payload: dict):
        with self._cond:
            self._payload = payload
            self._cond.notify()

    def configure(self, driver: str, circuit: str | None, n_sims: int | None = None):
        """New driver / circuit / precision: start a new engineer (computes at the next payload)."""
        with self._cond:
            self._config = (driver, circuit, n_sims or self.n_sims)
            self._cond.notify()

    def run(self):
        engineer = None
        while True:
            with self._cond:
                while self._running and self._payload is None and self._config is None:
                    self._cond.wait()
                if not self._running:
                    return
                payload, config = self._payload, self._config
                self._payload = self._config = None
            if config:
                driver, circuit, self.n_sims = config
                engineer = RaceEngineer(driver, circuit, n_sims=self.n_sims) if driver else None
                payload = payload or self._last
            if payload is None:
                continue
            self._last = payload
            if engineer is None:
                continue
            mine = completed_laps(payload, engineer.driver)
            new_lap = bool(mine) and mine[-1][0] > engineer.last_lap_done
            if new_lap:
                self.busy.emit(True)
            t0 = time.perf_counter()
            try:
                call = engineer.update(payload)
            except Exception:                                    # keep the window alive, show why
                self.failed.emit(traceback.format_exc(limit=3))
                call = None
            if new_lap:
                self.busy.emit(False)
            if call:
                self.call_ready.emit(call, time.perf_counter() - t0)

    def stop(self):
        with self._cond:
            self._running = False
            self._cond.notify()
        self.wait(5000)


# ── widgets ──────────────────────────────────────────────────────────────────
def panel(*widgets, caption: str | None = None) -> QFrame:
    f = QFrame()
    f.setObjectName("panel")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(12, 10, 12, 12)
    lay.setSpacing(6)
    if caption:
        c = QLabel(caption)
        c.setObjectName("caption")
        lay.addWidget(c)
    for w in widgets:
        lay.addWidget(w)
    return f


def table(headers: list[str]) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setSelectionMode(QAbstractItemView.NoSelection)
    t.setFocusPolicy(Qt.NoFocus)
    t.setShowGrid(False)
    t.verticalHeader().setDefaultSectionSize(24)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    t.horizontalHeader().setStretchLastSection(True)
    return t


def cell(text, colour: str | None = None, bold: bool = False, align=Qt.AlignLeft, italic=False) -> QTableWidgetItem:
    it = QTableWidgetItem(str(text))
    it.setTextAlignment(int(align | Qt.AlignVCenter))
    if colour:
        it.setForeground(QColor(colour))
    if bold or italic:
        f = QFont()
        f.setBold(bold)
        f.setItalic(italic)
        it.setFont(f)
    return it


def fmt_plan(plan) -> str:
    return " · ".join(f"L{lap}→{c[0]}" for lap, c in plan) if plan else "—"


class RaceEngineerWindow(QMainWindow):
    def __init__(self, host: str = "localhost", port: int = PORT, driver: str | None = None,
                 circuit: str | None = None, n_sims: int = 300):
        super().__init__()
        # Destroy the C++ window inside the event loop when it closes, not during interpreter
        # shutdown; and no lambdas capturing `self` in connections (they keep the window alive
        # until QApplication itself is torn down, which crashes in Qt's backing store).
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("F1 Race Replay — Race Engineer")
        self.resize(1180, 760)
        self.setStyleSheet(STYLE)
        self.driver, self.circuit_choice = (driver or "").upper() or None, circuit
        self.payload: dict | None = None
        self.last_call: dict | None = None
        self.messages = 0
        self._circuit_used: str | None = None
        self._take_control = False

        self._build_ui(n_sims)
        self.client = StreamClient(host, port)
        self.client.data_received.connect(self._on_payload)
        self.client.connection_status.connect(self._on_status)
        self.client.error_occurred.connect(self._on_error)
        self.worker = EngineerWorker(n_sims)
        self.worker.call_ready.connect(self._on_call)
        self.worker.busy.connect(self._on_busy)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()
        self.client.start()
        self.refresh = QTimer(self)                     # live widgets at 4 Hz, whatever the stream rate
        self.refresh.timeout.connect(self._refresh_live)
        self.refresh.start(250)

    # ── layout ───────────────────────────────────────────────────────────────
    def _build_ui(self, n_sims: int):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(14, 12, 14, 8)
        outer.setSpacing(10)

        # top bar
        top = QHBoxLayout()
        title = QLabel("RACE ENGINEER")
        title.setObjectName("title")
        self.mode = QLabel("")
        self.mode.setObjectName("caption")
        self.driver_box = QComboBox()
        self.driver_box.setMinimumWidth(90)
        if self.driver:
            self.driver_box.addItem(self.driver)
        self.driver_box.currentTextChanged.connect(self._driver_changed)
        self.circuit_box = QComboBox()
        self.circuit_box.addItem("Auto")
        from tools.circuit_profiles import load_profiles
        for c in sorted(k for k in load_profiles() if not k.startswith("_")):
            self.circuit_box.addItem(f"{c} Grand Prix")
        if self.circuit_choice:
            i = self.circuit_box.findText(self.circuit_choice)
            if i < 0:
                self.circuit_box.addItem(self.circuit_choice)
                i = self.circuit_box.count() - 1
            self.circuit_box.setCurrentIndex(i)
        self.circuit_box.currentTextChanged.connect(self._on_circuit)
        self.sims = QSpinBox()
        self.sims.setRange(50, 2000)
        self.sims.setSingleStep(50)
        self.sims.setValue(n_sims)
        self.sims.setSuffix(" sims")
        self.sims.editingFinished.connect(self._reconfigure)
        top.addWidget(title)
        top.addSpacing(12)
        top.addWidget(self.mode)
        top.addStretch(1)
        for lab, w in (("DRIVER", self.driver_box), ("CIRCUIT", self.circuit_box), ("PRECISION", self.sims)):
            c = QLabel(lab)
            c.setObjectName("caption")
            top.addWidget(c)
            top.addWidget(w)
            top.addSpacing(8)
        outer.addLayout(top)

        # headline
        self.headline = QLabel("Waiting for the stream…")
        self.headline.setObjectName("headline")
        self.headline.setWordWrap(True)
        self.sub = QLabel("Start f1-race-replay with --telemetry, or the stand-in stream.")
        self.sub.setObjectName("sub")
        self.sub.setWordWrap(True)
        self.alert = QLabel("")
        self.alert.setWordWrap(True)
        self.alert.setVisible(False)
        head = panel(self.headline, self.sub, self.alert, caption="CALL")
        self.head_panel = head
        outer.addWidget(head)

        # options + tower
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        self.options = table(["Option", "P(win)", "P(podium)", "P(points)", "Exp. pos", "vs stay out"])
        self.options.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.options.horizontalHeader().setStretchLastSection(False)
        self.options.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.options.setMinimumHeight(24 * 7 + 30)
        self.options.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        note = QLabel("Every option is raced to the flag on the same simulated races, so the "
                      "difference is the option, not luck. 'vs stay out' = places gained ± 95 % CI; "
                      "an option is only called when its gain beats its CI.\n"
                      "Rivals follow the assumed plan in the timing tower. Known biases: one-stop "
                      "strategies look too good at high-degradation tracks; no fuel effect between "
                      "strategies; dry races only.")
        note.setObjectName("note")
        note.setWordWrap(True)
        self.tower = table(["Pos", "Driver", "Gap", "Int", "Tyre", "Laps", "Stops", "Next stop (assumed)"])
        self.tower.verticalHeader().setDefaultSectionSize(22)
        self.live = QLabel("")
        self.live.setObjectName("sub")
        opt_panel = panel(self.options, caption="OPTIONS — paired Monte Carlo")
        opt_panel.layout().addStretch(1)
        opt_panel.layout().addWidget(note)
        grid.addWidget(opt_panel, 0, 0)
        grid.addWidget(panel(self.tower, self.live, caption="TIMING AT THE END OF THE LAP  (~ estimated)"), 0, 1, 2, 1)

        # take control
        self.tc_buttons = []
        tc = QWidget()
        tc.setObjectName("bare")
        tcl = QHBoxLayout(tc)
        tcl.setContentsMargins(0, 0, 0, 0)
        for comp in ("SOFT", "MEDIUM", "HARD"):
            b = QPushButton(f"BOX → {comp}")
            b.setStyleSheet(f"QPushButton {{ border-left: 4px solid {TYRE_COLOUR[comp]}; }}")
            b.setProperty("compound", comp)
            b.clicked.connect(self._on_box_button)
            tcl.addWidget(b)
            self.tc_buttons.append(b)
        stay = QPushButton("STAY OUT")
        stay.clicked.connect(self._on_stay_button)
        tcl.addWidget(stay)
        self.tc_buttons.append(stay)
        self.tc_log = QListWidget()
        self.tc_log.setMaximumHeight(110)
        self.tc_state = QLabel("")
        self.tc_state.setObjectName("sub")
        self.tc_panel = panel(self.tc_state, tc, self.tc_log, caption="TAKE CONTROL")
        self.tc_panel.setVisible(False)
        grid.addWidget(self.tc_panel, 1, 0)
        grid.setColumnStretch(0, 6)
        grid.setColumnStretch(1, 5)
        grid.setRowStretch(0, 1)
        outer.addLayout(grid, 1)

        self.setStatusBar(QStatusBar())
        self.conn = QLabel("Disconnected")
        self.count = QLabel("Messages: 0")
        self.statusBar().addPermanentWidget(self.count)
        self.statusBar().addPermanentWidget(self.conn)

    # ── stream ───────────────────────────────────────────────────────────────
    def _on_error(self, message: str):
        self.statusBar().showMessage(message, 5000)

    def _on_failed(self, trace: str):
        self.statusBar().showMessage("Engineer error: " + trace.strip().splitlines()[-1], 10000)

    def _on_circuit(self, _text: str):
        self._reconfigure()

    def _on_status(self, status: str):
        colour = {"Connected": GOOD, "Connecting…": WARN}.get(status, ACCENT)
        self.conn.setText(status)
        self.conn.setStyleSheet(f"color: {colour}; font-weight: 700;")

    def _on_payload(self, payload: dict):
        first = self.payload is None
        self.payload = payload
        self.messages += 1
        if first:
            self._first_payload(payload)
        self.worker.submit(payload)

    def _first_payload(self, payload: dict):
        drivers = (payload.get("frame") or {}).get("drivers") or {}
        order = sorted(drivers, key=lambda d: drivers[d].get("position", 99))
        self.driver_box.blockSignals(True)
        self.driver_box.clear()
        self.driver_box.addItems(order)
        if self.driver in order:
            self.driver_box.setCurrentText(self.driver)
        elif order:
            self.driver = order[0]
            self.driver_box.setCurrentIndex(0)
        self.driver_box.blockSignals(False)
        self._reconfigure()

    def _driver_changed(self, d: str):
        if d and d != self.driver:
            self.driver = d
            self._reconfigure()

    def _reconfigure(self):
        choice = self.circuit_box.currentText()
        if choice == "Auto":
            circuit, source = guess_circuit(self.payload or {})
        else:
            circuit, source = choice, "chosen"
        self._circuit_used = circuit
        self.mode.setText(f"{circuit or 'Generic circuit'} · {source}".upper())
        self.headline.setText(f"Following {self.driver} — call when they cross the line" if self.driver
                              else "Waiting for the stream…")
        self.headline.setStyleSheet(f"color: {TEXT};")
        self.worker.configure(self.driver, circuit, self.sims.value())

    # ── calls ────────────────────────────────────────────────────────────────
    def _on_busy(self, busy: bool):
        if busy:
            self.sub.setText(self.sub.text().split("  ·  computing")[0] + "  ·  computing…")

    def _on_call(self, call: dict, seconds: float):
        if call.get("driver") != self.driver:
            return                                              # stale: driver changed meanwhile
        self.last_call = call
        h = call["headline"]
        colour = TEXT if h.startswith("STAY") else (WARN if "BOX" in h or "STOPS" in h else INFO)
        if not call["running"]:
            colour = DIM
        self.headline.setText(h)
        self.headline.setStyleSheet(f"color: {colour};")
        self.head_panel.setStyleSheet(f"QFrame#panel {{ border-left: 4px solid {colour}; }}")

        st = call["standings"]
        bits = [f"Lap {call['lap']}/{call['total_laps']}"]
        if call["running"] and len(st):
            me = st[st["Driver"] == self.driver]
            if len(me):
                bits.append(f"{self.driver} P{int(me['Pos'].iloc[0])}")
        if call.get("tyres"):
            ty = call["tyres"]
            old = ty["age"] - 1                                    # age = FastF1 TyreLife of the next lap
            bits.append(f"{ty['compound'].title()}, {old} lap{'s' if old != 1 else ''} old")
            bits.append("2-compound rule ✓" if ty["rule_met"] else "2-compound rule still to meet")
        status = {"4": "SAFETY CAR", "5": "RED FLAG", "6": "VSC", "7": "VSC ENDING"}.get(call.get("track_status", "1"))
        if status:
            bits.append(status)
        if call.get("estimated"):
            bits.append(f"{len(call['estimated'])} cars estimated")
        bits.append(f"{seconds:.1f} s")
        self.sub.setText("  ·  ".join(bits))

        a = call.get("alert")
        if a:
            col = WARN if a["verdict"] != "STAY OUT" else DIM
            self.alert.setText(f"UNDERCUT CHECK on {a['target']} — {a['gap']:.1f} s ahead: {a['verdict']}"
                               f"  ·  closest after the stop {a['min_gap']:.1f} s  ·  P(pass) {a['p_overtake']:.0%}"
                               + ("  ·  in DRS range" if a["in_drs_window"] else ""))
            self.alert.setStyleSheet(f"color: {col}; font-weight: 600;")
        self.alert.setVisible(bool(a))
        self._fill_options(call)
        self._fill_tower(call)

    def _fill_options(self, call: dict):
        t = call["options"]
        self.options.setRowCount(len(t))
        for r, row in enumerate(t.itertuples(index=False)):
            opt = row[0]
            best = r == 0 and call["headline"].upper().startswith(opt.upper())
            stay = opt.startswith("Stay out")
            gain, ci = row[5], row[6]
            colour = GOOD if best else (TEXT if stay else None)
            self.options.setItem(r, 0, cell(opt, colour, bold=best or stay))
            for c, v in enumerate((row[1], row[2], row[3]), 1):
                self.options.setItem(r, c, cell(f"{v:.0%}", align=Qt.AlignRight))
            self.options.setItem(r, 4, cell(f"{row[4]:.2f}", colour, bold=best, align=Qt.AlignRight))
            g = "—" if stay else f"{gain:+.2f} ± {ci:.2f}"
            gcol = DIM if stay or abs(gain) <= ci else (GOOD if gain > 0 else ACCENT)
            self.options.setItem(r, 5, cell(g, gcol, align=Qt.AlignRight))

    def _fill_tower(self, call: dict):
        st = call["standings"]
        plans = call.get("rival_plans") or {}
        self.tower.setRowCount(len(st))
        for r, row in enumerate(st.itertuples(index=False)):
            me = row.Driver == self.driver
            est = bool(getattr(row, "Estimated", False))
            col = ACCENT if me else None
            self.tower.setItem(r, 0, cell(int(row.Pos), col, bold=me, align=Qt.AlignRight))
            self.tower.setItem(r, 1, cell(("~" if est else "") + row.Driver, col, bold=me, italic=est))
            gap = "LEADER" if r == 0 else f"+{row.Gap:.1f}"
            self.tower.setItem(r, 2, cell(gap, col, align=Qt.AlignRight, italic=est))
            itv = "" if r == 0 else f"{row.Interval:.1f}"
            self.tower.setItem(r, 3, cell(itv, col, align=Qt.AlignRight, italic=est))
            self.tower.setItem(r, 4, cell(row.Compound[0], TYRE_COLOUR.get(row.Compound), bold=True, align=Qt.AlignCenter))
            self.tower.setItem(r, 5, cell(int(row.TyreAge), align=Qt.AlignRight))      # laps done on the set
            self.tower.setItem(r, 6, cell(int(row.Stops), align=Qt.AlignRight))
            self.tower.setItem(r, 7, cell(fmt_plan(plans.get(row.Driver)), DIM))

    # ── live bits (no engine) ────────────────────────────────────────────────
    def _refresh_live(self):
        self.count.setText(f"Messages: {self.messages}")
        p = self.payload
        if not p:
            return
        sd = p.get("session_data") or {}
        fd = ((p.get("frame") or {}).get("drivers") or {}).get(self.driver or "", {})
        if fd:
            tyre = tyre_name(fd.get("tyre")) or "?"
            pit = "  ·  IN PIT" if fd.get("in_pit") else ""
            self.live.setText(f"Now: lap {sd.get('lap', '?')}/{sd.get('total_laps', '?')} (leader {sd.get('leader', '?')})"
                              f"  ·  {self.driver} P{fd.get('position', '?')} on lap {fd.get('lap', '?')},"
                              f" {tyre.title()} (lap {int(fd.get('tyre_life') or 0)} on the set){pit}")
        tc = p.get("take_control")
        if tc and not self._take_control:
            self._take_control = True
            self.tc_panel.setVisible(True)
        if tc:
            active = bool(tc.get("active"))
            for b in self.tc_buttons:
                b.setEnabled(active and self.driver in (tc.get("drivers") or []))
            self.tc_state.setText(
                f"The race engine is in control since lap {tc['from_lap']} — box {self.driver} whenever you want"
                if active else f"Real race until lap {tc['from_lap']}, then the engine takes over")
            self.tc_state.setStyleSheet(f"color: {WARN if active else DIM}; font-weight: 600;")
            log = tc.get("log") or []
            if self.tc_log.count() != len(log) or (log and self.tc_log.item(self.tc_log.count() - 1).text() != log[-1]):
                self.tc_log.clear()
                self.tc_log.addItems(log)
                self.tc_log.scrollToBottom()

    def _on_box_button(self):
        self._command("box", self.sender().property("compound"))

    def _on_stay_button(self):
        self._command("stay_out")

    def _command(self, cmd: str, compound: str | None = None):
        msg = {"cmd": cmd, "driver": self.driver}
        if compound:
            msg["compound"] = compound
        if not self.client.send(msg):
            self.statusBar().showMessage("Not connected — command not sent", 4000)

    # ── shutdown ─────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        self.refresh.stop()
        self.client.stop()
        self.worker.stop()
        event.accept()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--driver", default=None, help="three-letter code, e.g. ALO (default: the leader)")
    ap.add_argument("--circuit", default=None, help='e.g. "Bahrain Grand Prix" (default: from the stream)')
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--sims", type=int, default=300, help="Monte Carlo races per option")
    args = ap.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    w = RaceEngineerWindow(args.host, args.port, args.driver, args.circuit, args.sims)
    app.aboutToQuit.connect(w.close)                 # stop the threads however the app ends
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
