# Social Sentiment Analysis

Classify social posts as negative, neutral, or positive against human labels.

**Start with the [case study](CASE_STUDY.md), [executed phase-two notebook](Social_Sentiment_Phase2.ipynb), or [current results report](results/phase2/REPORT.md).**

Accuracy **58.2%**, macro F1 **0.5587** on 12,284 human-labelled social posts; VADER accuracy 53.0%.

![Current reliability and bin support](results/phase2/reliability.png)

The new evaluation adds independent calibration, uncertainty intervals, three development-only stability checks,
and auditable error analysis. Use `python predict_tweets.py --data examples/black_friday_posts.csv`. These examples are authored illustrations. Reconstruct local error text by source ID after fetching the pinned TweetEval files. This project measures general historical social sentiment, rather than Black Friday public opinion.

```bash
python -m pip install -r requirements.txt
python fetch_tweets.py
python phase2.py
```

Use a virtual environment; see [CASE_STUDY.md](CASE_STUDY.md) for the complete setup and interpretation.
The scripts use local project caches. Every original tracked file is retained. See [CHANGELOG.md](CHANGELOG.md)
and [AUDIT/phase2-file-changes.json](AUDIT/phase2-file-changes.json) for this pass’s changes.

<details>
<summary>Preserved phase-one benchmark and reproduction guide</summary>

# Sentiment Analysis

Classify English text sentiment and evaluate a compact model against independent human labels; also score unlabelled posts with VADER.

**Start with the [executed notebook](blackFriday.ipynb) or the [results report](results/REPORT.md).**
The report includes held-out predictions, a baseline, a precision–recall curve, confusion counts,
and the assumptions needed to interpret the results.

## Measured result

The validation-selected **tfidf_logistic** achieves test average precision **0.8873**,
precision **81.1%**, recall **76.8%**, and F1 **0.7889**.
Test log loss is **0.4730**; baseline log loss is **0.6931**.
These are the recorded benchmark results from the supplied input, not a deployment claim.

Review sentences differ from Black Friday social posts. This benchmark cannot validate neutral classification, establish a Black Friday sentiment distribution, or prove robustness to sarcasm or language shifts. The eight Black Friday example posts are authored illustrations, not collected observations.

## Run locally

Tested with Python **3.13.2**. Use Python 3.12 or newer and the pinned requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python fetch_data.py
python analysis.py
```

After the documented public input is present, the analysis runs locally without cloud services or credentials.
The downloader verifies checksums and leaves a differing existing input untouched.
For your own labelled CSV, use `python analysis.py --data path/to/input.csv --output results/custom_run`.
The data must match the [documented schema](data/README.md).

For predictions on new unlabelled rows, first run training, then:

```bash
python predict.py --data examples/new_rows.csv --output results/new_predictions.csv
```

The model bundle contains the fitted preprocessing and the validation-selected threshold.
Positive-score meanings: 0=negative; 1=positive. Score scales reflect the training benchmark.

Score unlabelled posts while retaining a neutral category:

```bash
python analysis.py --score-only --data examples/black_friday_posts.csv --output results/example_run
```

## Data and modelling choices

3,000 human-labelled sentences: 1,000 each from Amazon, IMDb, and Yelp, balanced within each source. Labels are 0=negative and 1=positive; there is no human-labelled neutral class. The original Black Friday tweets on Databricks/S3 were unavailable. The local tweet sample found on the Mac had no human labels and was left untouched. It is not published in this repository. IMDb contains multiline sentences; the converter parses the final tab-separated label and normalizes whitespace without losing rows.

Remove normalized duplicate text and conflicting labels before a stratified train/validation/test split. Fit TF-IDF unigrams/bigrams only on training text, preserving negation. Compare logistic regression with a prevalence baseline. Evaluate VADER separately against the held-out human labels, with neutral predictions explicitly counted as abstentions.

Sources, licence, sampling, schema, and SHA-256 checksums are in [data/README.md](data/README.md).

## Reviewer map

| File | What it demonstrates |
|---|---|
| `analysis.py` | Input validation, preprocessing, bounded model comparison, held-out evaluation |
| `workflow.py` | Split isolation, validation-only decisions, metrics, report generation |
| `predict.py` | Batch inference using the saved pipeline |
| `blackFriday.ipynb` | Executed walkthrough with current outputs |
| `results/REPORT.md` | Findings and limitations |
| `results/metrics.csv` | Validation and test metrics for every candidate |
| `results/predictions.csv` | Auditable held-out scores and labels |
| `results/splits.csv` | Split membership for every analysed row |
| `results/metrics.json` | Input hash, package versions, seed, and run settings |
| `tests/` | Input handling, leakage controls, metric correctness |

## Verification

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The notebook and CLI call the same implementation. Fixed seeds, pinned dependencies, and recorded input hashes
make the published run reproducible. Generated model files are local and ignored by Git.
See [REVIEW_NOTES.md](REVIEW_NOTES.md) for the repairs and remaining limits.


</details>
