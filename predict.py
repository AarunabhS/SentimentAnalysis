"""Batch inference with the pipeline and validation-selected threshold from analysis.py."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from analysis import ROOT, prepare_features


def predict(data: Path, model_path: Path, output: Path):
    frame = pd.read_csv(data)
    bundle = joblib.load(model_path)
    scores = bundle["model"].predict_proba(prepare_features(frame))[:, 1]
    result = pd.DataFrame({"row_id": range(len(frame)), "positive_score": scores,
                           "predicted": (scores >= bundle["threshold"]).astype(int)})
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, float_format="%.8f")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "results/model.joblib")
    parser.add_argument("--output", type=Path, default=ROOT / "results/new_predictions.csv")
    args = parser.parse_args()
    try:
        result = predict(args.data, args.model, args.output)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Input/model error: {error}\n")
    print(f"Scored {len(result)} rows: {args.output}")


if __name__ == "__main__":
    main()
