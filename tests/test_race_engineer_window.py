"""
Smoke test of the PySide6 Race Engineer window, offscreen: real socket, real
worker thread, synthetic race. Skipped when PySide6 is not installed.
"""
import os
import time

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from integrations.f1_race_replay.race_engineer_window import RaceEngineerWindow, guess_circuit  # noqa: E402
from integrations.f1_race_replay.replay_stream import RaceFeed, StreamServer, TakeControlFeed  # noqa: E402

from race_factory import RACE, line, make_race  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def pump(app, server, feed, t, until, dt=2.0, timeout=30.0):
    """Broadcast payloads (advancing the replay clock) and process Qt events until `until()`."""
    end = time.time() + timeout
    while time.time() < end:
        server.broadcast(feed.payload(t))
        t += dt
        for _ in range(4):
            app.processEvents()
            time.sleep(0.01)
        if until():
            return t
    raise AssertionError("timed out")


def test_window_shows_a_call_from_the_stream(app):
    feed = RaceFeed(make_race(20), 2099, RACE)
    server = StreamServer(port=0)
    w = RaceEngineerWindow(port=server.port, driver="D04", n_sims=20)
    try:
        w.show()
        pump(app, server, feed, line(feed, "D04", 4) - 10,
             lambda: w.last_call is not None and w.last_call["lap"] >= 5)
        assert w.driver_box.count() == 20 and w.driver_box.currentText() == "D04"
        assert w.headline.text().startswith(("STAY OUT", "BOX", "NO MORE STOPS"))
        assert w.tower.rowCount() == 20 and w.options.rowCount() >= 3
        assert "Bahrain Grand Prix" in w.mode.text().title()
        # switching driver re-targets the engineer
        w.driver_box.setCurrentText("D10")
        pump(app, server, feed, line(feed, "D10", 6) - 5,
             lambda: w.last_call is not None and w.last_call["driver"] == "D10")
        me = [w.tower.item(r, 1).text() for r in range(w.tower.rowCount())]
        assert any(x.endswith("D10") for x in me)
    finally:
        w.close()
        server.close()
    assert not w.client.isRunning() and not w.worker.isRunning()


def test_take_control_buttons_send_commands(app):
    feed = TakeControlFeed(make_race(20), 2099, RACE, 8, seed=1)
    server = StreamServer(port=0)
    w = RaceEngineerWindow(port=server.port, driver="D03", n_sims=10)
    try:
        w.show()
        pump(app, server, feed, feed.T0 + 30, lambda: w.tc_panel.isVisible() and w.tc_buttons[2].isEnabled())
        w.tc_buttons[2].click()                                   # BOX → HARD
        assert server.commands.get(timeout=3) == {"cmd": "box", "driver": "D03", "compound": "HARD"}
    finally:
        w.close()
        server.close()


def test_circuit_guess():
    assert guess_circuit({"session_info": {"event_name": "Monaco Grand Prix"}})[0] == "Monaco Grand Prix"
    assert guess_circuit({"circuit_length_m": 6950.0})[0] == "Belgian Grand Prix"   # measured, not official
    assert guess_circuit({"circuit_length_m": 3320.0, "session_data": {"total_laps": 78}})[0] == "Monaco Grand Prix"
    name, why = guess_circuit({"circuit_length_m": 5400.0, "session_data": {"total_laps": 57}})
    assert name is None and "Bahrain" in why and "Miami" in why    # same length, same laps
    assert guess_circuit({})[0] is None


def test_closing_waits_for_a_call_in_progress(app, monkeypatch):
    """Destroying a QThread that is still computing aborts the process: stop() must wait it out."""
    from integrations.f1_race_replay.race_engineer_window import EngineerWorker
    from tools.race_engineer import RaceEngineer

    def slow_update(self, payload):
        time.sleep(6.0)                                           # longer than any fixed timeout we had (5 s)
        return None

    monkeypatch.setattr(RaceEngineer, "update", slow_update)
    w = EngineerWorker(10)
    w.start()
    w.configure("D01", RACE)
    w.submit({"lap_times": {}})
    time.sleep(0.3)                                               # the slow call is running now
    w.stop()
    assert not w.isRunning()
