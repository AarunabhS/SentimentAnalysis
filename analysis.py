"""Evaluate sentiment against human labels and score unlabelled posts with VADER."""
from __future__ import annotations
import argparse
import re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from workflow import (SEED, binary_labels, input_metadata, json_write, make_splits,
                      require_columns, run_experiment, write_report)

ROOT = Path(__file__).resolve().parent


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, ["text"])
    if frame.text.isna().any() or not frame.text.map(lambda v: isinstance(v, str)).all():
        raise ValueError("Text values must be non-empty strings, not missing values or numbers.")
    text = frame.text.map(lambda value: re.sub(r"\s+", " ", value).strip())
    if text.eq("").any():
        raise ValueError("Text values must not be blank.")
    # Keep negation, emoji, and punctuation for sentiment interpretation.
    return pd.DataFrame({"text": text})


def vader_label(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def score_texts(frame: pd.DataFrame) -> pd.DataFrame:
    text = prepare_features(frame).text
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(value) for value in text]
    result = pd.DataFrame(scores)
    result.insert(0, "text", text.to_numpy())
    result["sentiment"] = result.compound.map(vader_label)
    return result


def load_data(path: Path):
    frame = pd.read_csv(path)
    require_columns(frame, ["text", "label"])
    frame["text"] = prepare_features(frame).text
    binary_labels(frame.label, "label")
    initial = len(frame)
    frame["key"] = frame.text.str.casefold()
    conflicts = frame.groupby("key").label.transform("nunique") > 1
    ambiguous = int(conflicts.sum())
    frame = frame.loc[~conflicts].drop_duplicates("key").reset_index(drop=True)
    return prepare_features(frame), binary_labels(frame.label, "label"), frame, {
        "rows_read": initial, "rows_used": len(frame), "duplicates_removed": initial-len(frame)-ambiguous,
        "conflicting_rows_removed": ambiguous, "split_strategy": "stratified 60/20/20 after normalized-text deduplication",
        "label_source": "human sentiment labels from UCI 331; never VADER-generated training labels"}


def sentiment_pipeline():
    text = ColumnTransformer([("text", TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=12000, sublinear_tf=True,
        strip_accents="unicode"), "text")])
    return make_pipeline(text, LogisticRegression(C=2.0, max_iter=600, random_state=SEED))


def run(data: Path = ROOT / "data/reviews.csv", output: Path = ROOT / "results"):
    data, output = Path(data), Path(output)
    X, y, frame, info = load_data(data)
    splits = make_splits(y)
    summary, model, predictions = run_experiment(X, y, {
        "baseline": DummyClassifier(strategy="prior"), "tfidf_logistic": sentiment_pipeline()},
        output, {"title": "Sentiment Analysis", "class_names": ["Negative", "Positive"], **input_metadata(data), **info}, splits=splits)
    joblib.dump({"model": model, "threshold": summary["selected_threshold"]}, output / "model.joblib")
    test_frame = frame.iloc[splits["test"]].reset_index(drop=True)
    # Independently measure VADER on human-labelled held-out sentences.
    rules = score_texts(test_frame)
    mapped = rules.sentiment.map({"negative": 0, "positive": 1})
    covered = mapped.notna()
    comparison = {
        "test_rows": len(test_frame), "coverage": float(covered.mean()),
        "neutral_abstentions": int((~covered).sum()),
        "accuracy_on_non_neutral_predictions": float((mapped[covered].to_numpy() == test_frame.label[covered].to_numpy()).mean()) if covered.any() else None,
        "correct_predictions_as_fraction_of_all_test_rows": float(((mapped == test_frame.label) & covered).mean()),
        "interpretation": "Neutral predictions abstain on this binary benchmark; neutral is never silently mapped to positive.",
    }
    json_write(output / "vader_benchmark.json", comparison)
    predictions["text"] = test_frame.text.to_numpy()
    predictions["source"] = test_frame.get("source", pd.Series("custom", index=test_frame.index)).to_numpy()
    predictions["vader_sentiment"] = rules.sentiment.to_numpy()
    predictions.to_csv(output / "predictions.csv", index=False, float_format="%.8f")
    predictions.loc[predictions.actual != predictions.predicted].to_csv(output / "errors.csv", index=False, float_format="%.8f")
    predictions.groupby("source").apply(lambda part: pd.Series({
        "test_rows": len(part), "accuracy": float((part.actual == part.predicted).mean())}),
        include_groups=False).to_csv(output / "source_breakdown.csv", float_format="%.8f")
    examples = ROOT / "examples/black_friday_posts.csv"
    if examples.exists():
        score_texts(pd.read_csv(examples)).to_csv(output / "example_sentiments.csv", index=False)
    write_report(output, summary,
        "Benchmark positive/negative sentiment against UCI's human-labelled review sentences. "
        "Training-only TF-IDF and logistic regression are evaluated against an independent holdout. "
        "VADER additionally scores unlabelled social posts as positive, negative, or neutral.",
        f"The VADER baseline abstained as neutral on {comparison['neutral_abstentions']} of {len(test_frame)} "
        f"held-out sentences (coverage {comparison['coverage']:.1%}); see `vader_benchmark.json` for accuracy "
        "with its denominator explicitly stated. Training a classifier to reproduce VADER labels would measure "
        "agreement with a rule system, not accuracy against human sentiment. This workflow instead uses human labels. "
        "`errors.csv` exposes model failures, and `source_breakdown.csv` separates Amazon, IMDb, and Yelp performance. "
        "The benchmark intentionally contains only positive/negative sentences, so it cannot validate neutral detection. "
        "Reviews differ from Black Friday tweets; no claim of measured Black Friday public opinion is made. "
        "`example_sentiments.csv` scores eight authored illustrative posts, not collected social-media observations. "
        "Sarcasm, mixed sentiment, language shifts, and domain shifts remain limitations.")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/reviews.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--score-only", action="store_true", help="Score an unlabelled CSV with a text column using VADER.")
    args = parser.parse_args()
    try:
        if args.score_only:
            scores = score_texts(pd.read_csv(args.data))
            args.output.mkdir(parents=True, exist_ok=True)
            scores.to_csv(args.output / "sentiments.csv", index=False)
            print(f"Scored {len(scores)} texts: {args.output / 'sentiments.csv'}")
        else:
            result = run(args.data, args.output)
            print(f"Selected {result['selected_model']}; test F1={result['test']['f1']:.4f}; report: {args.output / 'REPORT.md'}")
    except (ValueError, OSError) as error:
        parser.exit(2, f"Input error: {error}\n")


if __name__ == "__main__":
    main()
