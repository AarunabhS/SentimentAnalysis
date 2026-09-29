# Data source and schema

Source: [Dimitrios Kotzias, UCI Sentiment Labelled Sentences (2015)](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences).

Licence: **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Dataset rights are separate from project code.

3,000 human-labelled sentences: 1,000 each from Amazon, IMDb, and Yelp, balanced within each source. Labels are 0=negative and 1=positive; there is no human-labelled neutral class. The original Black Friday tweets on Databricks/S3 were unavailable. The local tweet sample found on the Mac had no human labels and was left untouched. It is not published in this repository. IMDb contains multiline sentences; the converter parses the final tab-separated label and normalizes whitespace without losing rows.

## Schema

CSV columns: `text, label, source`.

Target: `label` — 0=negative; 1=positive.
Extra columns are excluded by the explicit feature list in `analysis.py`.
Missing or non-binary labels are rejected, never inferred as negatives.
Missing or blank review text is rejected before splitting.

## Reproduce and verify

`python fetch_data.py` verifies the local file or downloads from the exact source above if it is missing.
Both the upstream bytes and the converted CSV are checked against the SHA-256 hashes in
[provenance.json](provenance.json). Existing differing files are left untouched.
CSV conversions are deterministic with the tested requirements.

See the [results report](../results/REPORT.md) for the actual cohort, partition sizes, and limitations.
