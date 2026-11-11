# Draft: proposals for IAmTomShaw/f1-race-replay

**Status: ready to post — not submitted.** They go out under Paco's name, so he posts them by hand
(copy the title and body). The project's guidelines ask to discuss in an issue before a PR.
Order: **Issue A** first (a bug, easy to accept), then **Issue B** (a small feature, with a PR offer).
Checked on 11 Nov 2026 against upstream `9b3e39f`: neither is reported yet. Related: #354.

---

## Issue A — bug

**Title:** Cached sessions still fail offline: `get_circuit_rotation()` needs the MultiViewer API

**Body:**

Related to #354. Since F1 started answering 401 to unauthenticated timing downloads, the only races
that can be replayed are the ones already in the FastF1 cache. Those load fine with FastF1's offline
mode — every timing, car and position file comes from the cache — but the replay still crashes before
the window opens, because `get_circuit_rotation()` calls `session.get_circuit_info()`, which always
requests `https://api.multiviewer.app/api/v1/circuits/<key>/<year>` and is not cached.

**Steps to reproduce**

1. Have a race in the FastF1 cache with telemetry (e.g. 2023 Bahrain GP, loaded before October 2026).
2. Enable FastF1 offline mode (`fastf1.Cache.offline_mode(True)`) or run without network.
3. `python main.py --viewer --year 2023 --round 1`

**Expected:** the replay opens from the cache.
**Actual:** an exception from `get_circuit_info()` (`AttributeError: 'NoneType' object has no attribute
'decode'` in offline mode; a request error without network). `main.py` line 144 calls it unguarded.

**Suggested fix** (`src/f1_data.py`): fall back to a rotation computed from the track itself, so the map
is still drawn sensibly:

```python
def get_circuit_rotation(session):
    try:
        return session.get_circuit_info().rotation
    except Exception:
        # No MultiViewer data (offline / API down): make the track's longest axis horizontal
        pos = session.laps.pick_fastest().get_pos_data()
        xy = np.column_stack([pos["X"], pos["Y"]]).astype(float)
        xy -= xy.mean(axis=0)
        vx, vy = np.linalg.eigh(np.cov(xy.T))[1][:, -1]
        return -float(np.degrees(np.arctan2(vy, vx)))
```

Tested on 2023 Bahrain, 2023 Saudi Arabia, 2023 Australia and 2024 Saudi Arabia from a local cache with
offline mode on: the replay and the telemetry stream work end to end. Happy to open a PR.

---

## Issue B — feature

**Title:** Telemetry stream: include the session (event name, year, round) in the payload

**Body:**

I'm building a pit-wall window on top of the telemetry stream (a race-strategy assistant that makes
box / stay-out calls for a chosen driver while the replay runs). The stream is great for this — the
`lap_times` with `replay_*_time_s`, `status_laps` and the frame data are everything a live tool needs —
except for one thing: **a client can't tell which race is playing.**

Anything circuit-specific (pit-loss time, Safety Car likelihood, overtaking difficulty, degradation)
needs the event. Today a client can only guess from `circuit_length_m` and `session_data.total_laps`,
and that is ambiguous for several tracks (Bahrain and Miami are both ~5.41 km and 57 laps; Monza and
Suzuka are both ~5.8 km and 53 laps; Yas Marina and Albert Park too). I checked it against the real
payloads of the current version: there is no other field that identifies the race.

`F1RaceReplayWindow` already receives this information as `session_info` (used for the HUD banner), so
the change is small and backwards compatible: add it to the broadcast.

Proposed field:

```json
"session_info": {
  "event_name": "Bahrain Grand Prix",
  "circuit_name": "Bahrain",
  "country": "Bahrain",
  "year": 2023,
  "round": 1,
  "date": "March 05, 2023"
}
```

Happy to open the PR (code + `telemetry.md` docs + a test) if this sounds good.

---

## Pull request

**Title:** Telemetry stream: add `session_info` to the broadcast payload

**Body:**

Closes #<issue>.

Clients of the telemetry stream can't identify the race being replayed, which circuit-specific insight
windows need. `F1RaceReplayWindow` already has `session_info`; this PR adds its identifying fields to
every broadcast.

- `src/interfaces/race_replay.py`: keep a JSON-safe copy of the session fields in `__init__`, add it to
  the payload in `_broadcast_telemetry_state()` when available. A few dozen bytes per message; existing
  clients ignore the new key.
- `telemetry.md`: document the field.
- `tests/`: payload test for the new key (and that it is absent when no `session_info` was given).

No behaviour change for the replay window or the existing insight windows.

### Diff (sketch)

```diff
--- a/src/interfaces/race_replay.py
+++ b/src/interfaces/race_replay.py
@@ def __init__(self, frames, track_statuses, example_lap, drivers, title,
         self.session_info_comp = SessionInfoComponent(visible=visible_hud)
         self.circuit_length_m = session_info.get('circuit_length_m') if session_info else None
+        # Identifies the replayed session for telemetry-stream clients (JSON-safe subset)
+        self._stream_session_info = (
+            {k: session_info.get(k) for k in ("event_name", "circuit_name", "country", "year", "round", "date")}
+            if session_info else None
+        )
         if session_info:
             self.session_info_comp.set_info(
@@ def _broadcast_telemetry_state(self):
             "session_data": {
                 "time": time_str,
                 "time_s": float(t),
                 "lap": leader_lap,
                 "leader": leader_code,
                 "total_laps": self.total_laps
             }
         }
+        if getattr(self, "_stream_session_info", None):
+            payload["session_info"] = self._stream_session_info
```

```diff
--- a/telemetry.md
+++ b/telemetry.md
@@ ## Telemetry Data Format
+### Session Info
+
+`session_info` identifies the replayed session (absent if the replay was started without it):
+
+| Field | Type | Description |
+|-------|------|-------------|
+| `event_name` | string | FastF1 `EventName`, e.g. `"Bahrain Grand Prix"` |
+| `circuit_name` | string | Circuit name shown in the HUD |
+| `country` | string | Event country |
+| `year` | int | Season |
+| `round` | int | Round number |
+| `date` | string | Event date, e.g. `"March 05, 2023"` |
```

### Notes for the maintainer

- `round` is whatever `main.py` passes (`round_number`); for sprint sessions the session type could
  also be added (`"session_type": "R" | "S"`) — left out to keep the change minimal.
- The Race Engineer window that motivated this works with the current payload too (it falls back to
  guessing the circuit, or asks the user); with `session_info` it needs no input.
