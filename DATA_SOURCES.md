# Data provenance and licensing

Source: CDC U.S. Outpatient Influenza-like Illness Surveillance Network (ILINet), served by CMU Delphi. The source is identified as **Public Domain US Government** in Delphi's official documentation (checked 2026-10-20). Retain attribution when redistributing this snapshot.

- Definitions, revision behavior, geography and license: https://cmu-delphi.github.io/delphi-epidata/api/v5-signals/fluview_ilinet.html
- Actual legacy V3 query: https://api.delphi.cmu.edu/epidata/fluview/?regions=hhs1,hhs2,hhs3,hhs4,hhs5,hhs6,hhs7,hhs8,hhs9,hhs10&epiweeks=201040-201939
- Legacy endpoint documentation and V5 migration: https://cmu-delphi.github.io/delphi-epidata/api/fluview.html
- CDC historical-release information: https://cdcepi.github.io/flu/data.html
- CDC/MMWR calendar implementation: https://epiweeks.readthedocs.io/en/stable/usage.html

The real raw response is `data/raw/fluview_201040_201939.json`; URL, HTTP status, API result/message, acquisition UTC timestamp, exact byte count, field list, row count, and SHA-256 are recorded in `data/source_manifest.json`. It was fetched in this session using curl; the manifest acquisition time is the completed file's filesystem modification time, expressed in UTC. The reusable loader uses Python urllib when a snapshot is absent.

No credentials or auth parameter were used. No V5 fallback or field remapping was required. Original `release_date`, `issue`, and other upstream metadata remain in the raw snapshot; normalized data retain them too. Models receive only wili history, region indicators and target-date annual features. The retrospective snapshot contains revised observations; release metadata do not establish a real-time information set.

`wili` is a population-weighted percentage of outpatient visits with ILI. For example 2.5 means 2.5%; errors are percentage points. It is not a population infection rate or lab-confirmed influenza. Data are not interpolated or smoothed by this project. All ten regions have a complete weekly panel and genuine week 53 is retained.
