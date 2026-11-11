"""
Smoke test of the PySide6 Race Engineer window, offscreen: real socket, real
worker thread, synthetic race. Skipped when PySide6 is not installed.
"""
import os
import time

import numpy as np

import pytest

try:
    import PySide6.QtWidgets  # noqa: F401
except ImportError:            # not installed, or a broken install (e.g. a DLL that won't load on Windows)
    pytest.skip("PySide6 not available", allow_module_level=True)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from integrations.f1_race_replay.race_engineer_window import RaceEngineerWindow, guess_circuit  # noqa: E402
from integrations.f1_race_replay.replay_stream import RaceFeed, ReplaySession, StreamServer, TakeControlFeed  # noqa: E402

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
        pump(app, server, feed, feed.T0 + 30, lambda: w.tc_buttons[2].isEnabled())
        assert not w.take_button.isEnabled() and "IN CONTROL" in w.take_button.text()
        w.tc_buttons[2].click()                                   # BOX → HARD
        assert server.commands.get(timeout=3) == {"cmd": "box", "driver": "D03", "compound": "HARD"}
    finally:
        w.close()
        server.close()


def test_take_control_button_hands_the_race_to_the_engine(app):
    """Real race first; the button sends take_control, the stream switches, BOX buttons come alive."""
    session = ReplaySession(make_race(20), 2099, RACE, seed=1)
    server = StreamServer(port=0)
    w = RaceEngineerWindow(port=server.port, driver="D03", n_sims=10)

    class Relay:                                                   # what replay_stream.main() does
        def payload(self, t):
            while not server.commands.empty():
                session.command(server.commands.get(), t)
            return session.payload(t)
    try:
        w.show()
        t = pump(app, server, Relay(), line(session.feed, "D00", 5) + 10, lambda: w.take_button.isEnabled())
        assert not any(b.isEnabled() for b in w.tc_buttons)       # real race: no BOX
        assert "Real race" in w.tc_state.text()
        w.take_button.click()
        pump(app, server, Relay(), t, lambda: w.tc_buttons[0].isEnabled())
        assert session.in_control and session.feed.N >= 5
        assert "IN CONTROL" in w.tc_state.text() and not w.take_button.isEnabled()
    finally:
        w.close()
        server.close()


def test_take_control_is_off_with_the_real_f1_race_replay(app):
    """f1-race-replay's payload has no take_control: the button stays off and says why."""
    feed = RaceFeed(make_race(20), 2099, RACE)
    server = StreamServer(port=0)
    w = RaceEngineerWindow(port=server.port, driver="D03", n_sims=10)
    try:
        w.show()
        pump(app, server, feed, line(feed, "D00", 3), lambda: "stand-in stream" in w.tc_state.text())
        assert not w.take_button.isEnabled() and not any(b.isEnabled() for b in w.tc_buttons)
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


def test_track_map_draws_the_stand_in_race_on_the_circuit(app):
    """TRACK MAP opens a second window fed by the same stream; cars sit on the GPS outline."""
    from integrations.f1_race_replay.track_view import TrackMap
    feed = RaceFeed(make_race(20), 2099, RACE)
    p = feed.payload(line(feed, "D00", 3) + 30)
    assert p["position_source"] == "lap-fraction" and len(p["track_geometry"]["x"]) == 400
    d = p["frame"]["drivers"]["D00"]
    gx, gy = np.asarray(p["track_geometry"]["x"]), np.asarray(p["track_geometry"]["y"])
    assert np.min(np.hypot(gx - d["x"], gy - d["y"])) < 300          # on the outline (dm), not on a circle
    m = TrackMap("D00")
    m.resize(700, 500)
    m.set_payload(p)
    m.set_payload(feed.payload(line(feed, "D00", 3) + 34))
    img = m.grab().toImage()                                         # paints without errors
    assert img.width() == 700 and m.geometry_xy is not None and len(m._frac) == 20
    server = StreamServer(port=0)
    w = RaceEngineerWindow(port=server.port, driver="D00", n_sims=10)
    try:
        w.show()
        w.open_track_map()
        pump(app, server, feed, line(feed, "D00", 2), lambda: w.track_map.payload is not None)
        assert w.track_window.isVisible()
    finally:
        w.close()
        server.close()


def test_before_the_first_call_the_window_says_when_it_comes(app):
    """Lap 1 of the real replay: connected, no call yet — the window says the first call is at lap 3."""
    feed = RaceFeed(make_race(20), 2099, RACE)
    server = StreamServer(port=0)
    w = RaceEngineerWindow(port=server.port, driver="D03", n_sims=10)
    try:
        w.show()
        pump(app, server, feed, 5.0, lambda: "first call when D03 completes lap 3" in w.sub.text(), dt=1.0)
        assert w.last_call is None
    finally:
        w.close()
        server.close()
