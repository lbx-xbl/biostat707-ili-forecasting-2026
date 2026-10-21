# BIOSTAT 707 ILI Forecasting Project

This repository contains our [project proposal](https://github.com/lbx-xbl/biostat707-ili-forecasting-2026/blob/main/707-proposal.md), the historical ILI data snapshot, pilot modeling code, and saved pilot results.

## Data source and citation

We use the population-weighted percentage of outpatient visits with ILI (`wili`) from the CDC U.S. Outpatient Influenza-like Illness Surveillance Network (ILINet), accessed through the [CMU Delphi FluView API](https://cmu-delphi.github.io/delphi-epidata/api/fluview.html). The [Delphi source documentation](https://cmu-delphi.github.io/delphi-epidata/api/v5-signals/fluview_ilinet.html) identifies these U.S. Government data as public domain. Please cite CDC ILINet and CMU Delphi when using the data.

## Download and check the data

From the repository root, run:

```bash
python3 scripts/download_ili_data.py
python3 scripts/download_ili_data.py --verify-only
```

The script requests weekly `wili` values for the ten U.S. HHS regions from 2010w40 through 2019w39. It checks the region-week coverage and values, then writes the response to `data/raw/` and a download record with a SHA-256 checksum to `data/source_manifest.json`. It requires no API key or third-party Python package.

The script uses Delphi's legacy FluView endpoint to match the pilot data format. Historical values can be revised, so a new download may not reproduce the pilot snapshot byte-for-byte. The legacy endpoint is also being phased out; any future switch to the newer API will need separate validation. The current analysis is retrospective and is not a verified real-time forecast.

## Reproduce the retrospective pilot

The saved pilot report is at `outputs/demo_v0_1/demo_report.html`. Its figures, predictions, and metrics are preserved in the same folder. The pilot used the committed historical data snapshot; it is not a verified real-time forecast.

The original demo was tested on macOS arm64 with Python 3.11.5. From the repository root, a new run can be attempted with:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_demo.py --config config/demo.json --output-dir outputs/demo_rerun
```

The runner checks the data snapshot and runs the tests before fitting the models. It refuses a nonempty output directory, so use a new directory name for each rerun. The code and saved outputs are now in this group repository, but a fresh-clone run from this repository has not yet been verified. Do not upload the local `.venv` folder.
