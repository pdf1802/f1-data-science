# Put the dashboard online (free)

The dashboard runs on a fresh clone because `demo/` ships what it reads (lap data, trained models, a few
qualifying and telemetry sessions). That is all a free host needs.

## Streamlit Community Cloud

1. Push the repository to GitHub (the `demo/`, `static/`, `ui/` and `.streamlit/` folders must be in the branch).
2. Go to <https://share.streamlit.io>, sign in with GitHub and choose **Create app → Deploy a public app from GitHub**.
3. Repository `pdf1802/f1-data-science`, branch `main`, main file `app.py`. Under *Advanced settings* pick
   Python 3.12. No secrets are needed.
4. Deploy. The first build installs `requirements.txt` (a few minutes). The header of the app shows
   **Mode: demo bundle**.
5. Put the URL at the top of `README.md` (there is a commented line waiting for it) and on your CV.

Nothing is downloaded at run time: FastF1 stays in offline mode and, with no local cache, every tab reads the bundle.

## What the demo can and cannot do

| Tab | In the demo |
|---|---|
| Race engineer, How many stops, Undercut, What-if | Everything (35 dry races, 2022–2025) |
| Race simulator, Qualifying → race | The qualifying sessions in `demo/qualifying.parquet` |
| Telemetry | The fastest laps in `demo/telemetry.parquet` (a few sessions, top 8 drivers) |

The Race Engineer **window** (PySide6) and the f1-race-replay integration are desktop programs: they are in the
README's GIF, not in the web demo.

## Refresh the bundle

On a machine with the lap Parquet, trained models and a FastF1 cache:

```bash
python -m tools.train_models --laps data/laps_2022_2025.parquet     # if the models changed
python -m tools.demo_data                                           # rewrites demo/
git add demo && git commit -m "Refresh the demo bundle"
```

To add telemetry sessions, edit `TELEMETRY_SESSIONS` in `tools/demo_data.py`.

## About the data

The bundle contains timing data derived from the FastF1 cache (lap times, qualifying times, telemetry of a few
laps). It is there for an educational, non-commercial demo. Formula 1 timing data belongs to its owners; this
project is not affiliated with Formula 1, the FIA or any team. If you fork this for anything else, check the terms
that apply to you.
