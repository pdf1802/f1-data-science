# Draft: proposal for IAmTomShaw/f1-race-replay

**Status: draft only — not submitted.** Text for an issue first (the project's contribution guidelines
ask to discuss a feature in an issue before a PR), then the PR if the maintainer agrees.

---

## Issue

**Title:** Telemetry stream: include the session (event name, year, round) in the payload

**Body:**

I'm building a pit-wall window on top of the telemetry stream (a race-strategy assistant that makes
box / stay-out calls for a chosen driver while the replay runs). The stream is great for this — the
`lap_times` with `replay_*_time_s`, `status_laps` and the frame data are everything a live tool needs —
except for one thing: **a client can't tell which race is playing.**

Anything circuit-specific (pit-loss time, Safety Car likelihood, overtaking difficulty, degradation)
needs the event. Today a client can only guess from `circuit_length_m` and `session_data.total_laps`,
and that is ambiguous for several tracks (Bahrain and Miami are both ~5.41 km and 57 laps; Monza and
Suzuka are both ~5.8 km and 53 laps).

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
