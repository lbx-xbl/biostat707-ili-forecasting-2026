# BIOSTAT 707 ILI Forecasting Project

This repository contains our [project proposal](707-proposal.md) and a script for downloading the public influenza-like illness (ILI) data used in the project. Model and evaluation code will be added as the group develops the analysis.

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
