# Change log

## Phase two — 30 September 2026

- Added `phase2.py`, `review_tools.py`, executed `Social_Sentiment_Phase2.ipynb`, and `CASE_STUDY.md`.
- Added independent calibration, uncertainty estimates, development-only stability checks, and error analysis.
- Updated the README to lead with current evidence while preserving its phase-one content.
- Added focused tests. Preserved existing files, phase-one results, notebooks, and Git history.
- Use `python predict_tweets.py --data examples/black_friday_posts.csv`. These examples are authored illustrations. Reconstruct local error text by source ID after fetching the pinned TweetEval files. This project measures general historical social sentiment, rather than Black Friday public opinion.

Data provenance and download checks are documented in `data/`. File paths and before/after SHA-256 hashes
are recorded in `AUDIT/phase2-file-changes.json`. Commit and push receipts are retained in the local audit log.
