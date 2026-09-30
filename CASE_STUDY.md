# Social Sentiment Analysis: case study

## Problem and outcome

Classify social posts as negative, neutral, or positive against human labels.

Accuracy **58.2%**, macro F1 **0.5587** on 12,284 human-labelled social posts; VADER accuracy 53.0%.

## Data and evaluation

TweetEval’s SemEval-2017 sentiment task supplies 59,899 human-labelled posts. The original test split retains all 12,284 posts. Text-only overlap removal protects it from training contamination; ambiguous training labels and repeated training text are excluded. The upstream dataset licence is unspecified, so raw posts and error text stay local.

The three-way model is compared with a prior baseline and VADER. A separate diagnostic measures the old review model on binary-labelled tweets. The different datasets, classes, and prevalences mean that 58% social accuracy cannot be compared directly with 79.5% binary-review accuracy. Lexical negation, contrast, and sarcasm hints support inspection; they are not human annotations of those properties.

Model decisions use validation only. Preprocessing fits on fitting rows; probability calibration uses a disjoint reserve.
The test split is evaluated after model selection. Reusing known benchmarks during development is distinct from a new external validation.

## Findings, errors, and uncertainty

The [executed notebook](Social_Sentiment_Phase2.ipynb) displays results and uncertainty from the same Python implementation.
[Current report](results/phase2/REPORT.md), [reliability plot](results/phase2/reliability.png),
[validation stability](results/phase2/validation_stability.csv), and [uncertainty intervals](results/phase2/uncertainty.json)
show performance with its practical limits. [Error rows](results/phase2/errors.csv) expose failures rather than hiding them.

Use `python predict_tweets.py --data examples/black_friday_posts.csv`. These examples are authored illustrations. Reconstruct local error text by source ID after fetching the pinned TweetEval files. This project measures general historical social sentiment, rather than Black Friday public opinion.

## Reproduce

Tested with the phase-one pinned Python requirements. Use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python fetch_tweets.py
python phase2.py
```

Phase-one results and notebooks remain available. Phase-two outputs are written to `results/phase2/`.
Model bundles and locally retained raw inputs are ignored by Git. Data downloaders verify pinned hashes and preserve differing existing inputs.
Install `requirements-dev.txt` for `python -m pytest -q` or to execute the notebook.
