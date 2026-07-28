"""Evaluate frozen code representations with linear and nonlinear learners."""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer, T5EncoderModel
from xgboost import XGBClassifier

from frozen_codebert_baseline import encode_unique, load_pairs


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "authentic_only" / "datasets" / "analysis_dataset.parquet"
OUTPUT = ROOT / "outputs" / "frozen_representation_ablation"
LEGACY_CODEBERT = (
    ROOT
    / "outputs"
    / "frozen_codebert_baseline"
    / "pair_embeddings_n30979_l256.npz"
)
SEED = 42
BOOTSTRAPS = 500

MODELS = {
    "CodeBERT": {"checkpoint": "microsoft/codebert-base", "kind": "auto"},
    "GraphCodeBERT": {
        "checkpoint": "microsoft/graphcodebert-base",
        "kind": "auto",
    },
    "UniXcoder": {"checkpoint": "microsoft/unixcoder-base", "kind": "auto"},
    "CodeT5": {"checkpoint": "Salesforce/codet5-small", "kind": "t5"},
}


def status(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def clustered_interval(
    y: np.ndarray, probability: np.ndarray, groups: np.ndarray
) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    unique = np.unique(groups)
    positions = {group: np.flatnonzero(groups == group) for group in unique}
    estimates: list[float] = []
    for _ in range(BOOTSTRAPS):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([positions[group] for group in sampled])
        if np.unique(y[indices]).size == 2:
            estimates.append(roc_auc_score(y[indices], probability[indices]))
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def load_encoder(name: str, device: torch.device):
    configuration = MODELS[name]
    status(f"Downloading or loading {configuration['checkpoint']}")
    tokenizer = AutoTokenizer.from_pretrained(configuration["checkpoint"])
    model_class = T5EncoderModel if configuration["kind"] == "t5" else AutoModel
    model = model_class.from_pretrained(configuration["checkpoint"])
    model.to(device).eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    status(f"Loaded {name} with {parameters:,} parameters on {device}")
    return tokenizer, model


def embeddings(
    name: str,
    frame: pd.DataFrame,
    sources: list[str],
    candidates: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    slug = name.lower()
    path = OUTPUT / f"{slug}_pair_embeddings_n{len(frame)}_l{args.max_length}.npz"
    root_checkpoint = ROOT / "outputs" / path.name
    checkpoint = path if path.exists() else root_checkpoint
    if args.resume and checkpoint.exists():
        status(f"Restoring completed {name} embeddings from {checkpoint}")
        stored = np.load(checkpoint)
        return stored["source"], stored["candidate"], 0.0
    if name == "CodeBERT" and args.resume and LEGACY_CODEBERT.exists():
        status("Importing the completed CodeBERT embeddings from the prior run")
        stored = np.load(LEGACY_CODEBERT)
        source = stored["source"]
        candidate = stored["candidate"]
        np.savez_compressed(path, source=source, candidate=candidate)
        return source, candidate, 0.0

    tokenizer, model = load_encoder(name, device)
    model_sources = sources
    model_candidates = candidates
    if name == "UniXcoder":
        model_sources = [f"<encoder-only> {text}" for text in sources]
        model_candidates = [f"<encoder-only> {text}" for text in candidates]
    started = time.perf_counter()
    status(f"Encoding {name} source programs")
    source = encode_unique(
        model_sources, tokenizer, model, args.batch_size, args.max_length, device
    )
    status(f"Encoding {name} candidate programs")
    candidate = encode_unique(
        model_candidates, tokenizer, model, args.batch_size, args.max_length, device
    )
    seconds = time.perf_counter() - started
    status(f"Saving {name} embeddings after {seconds / 60:.1f} minutes")
    np.savez_compressed(path, source=source, candidate=candidate)
    del model, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return source, candidate, seconds


def pair_features(source: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    difference = np.abs(source - candidate)
    product = source * candidate
    cosine = np.sum(source * candidate, axis=1, keepdims=True) / (
        np.linalg.norm(source, axis=1, keepdims=True)
        * np.linalg.norm(candidate, axis=1, keepdims=True)
    ).clip(min=1e-12)
    return np.concatenate([difference, product, cosine], axis=1)


def main() -> None:
    args = arguments()
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pd.read_parquet(DATA)
    status(f"Loaded {len(frame):,} authentic source-candidate pairs")
    sources, candidates = load_pairs(frame)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_index, test_index = next(
        splitter.split(frame, frame["label"], groups=frame["problem_id"])
    )
    test = frame.iloc[test_index].copy()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    predictions = test[["example_id", "problem_id", "quality_score", "label"]].copy()
    timing = {}

    for representation in args.models:
        status(f"Starting representation {representation}")
        source, candidate, encoding_seconds = embeddings(
            representation, frame, sources, candidates, args, device
        )
        status(f"Constructing pair features for {representation}")
        features = pair_features(source, candidate)
        learners = {
            "Logistic Regression": make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, random_state=SEED),
            ),
            "XGBoost": XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                tree_method="hist",
                device="cuda" if device.type == "cuda" else "cpu",
                random_state=SEED,
            ),
        }
        timing[representation] = {"encoding_seconds": encoding_seconds}
        for learner_name, learner in learners.items():
            status(f"Training {representation} with {learner_name}")
            started = time.perf_counter()
            learner.fit(features[train_index], frame.iloc[train_index]["label"])
            fit_seconds = time.perf_counter() - started
            probability = learner.predict_proba(features[test_index])[:, 1]
            column = f"{representation}__{learner_name}"
            predictions[column] = probability
            timing[representation][f"{learner_name}_fit_seconds"] = fit_seconds
            status(
                f"Completed {representation} with {learner_name} in "
                f"{fit_seconds:.1f} seconds"
            )
            for scope, selected in {
                "All grades": np.ones(len(test), dtype=bool),
                "Score 2 vs 3": test["quality_score"].isin([2, 3]).to_numpy(),
            }.items():
                y = test.loc[selected, "label"].to_numpy()
                p = probability[selected]
                groups = test.loc[selected, "problem_id"].to_numpy()
                low, high = clustered_interval(y, p, groups)
                rows.append(
                    {
                        "representation": representation,
                        "learner": learner_name,
                        "scope": scope,
                        "n": len(y),
                        "roc_auc": roc_auc_score(y, p),
                        "roc_auc_ci_low": low,
                        "roc_auc_ci_high": high,
                        "pr_auc": average_precision_score(y, p),
                        "f1_at_0_5": f1_score(y, p >= 0.5),
                    }
                )
                status(
                    f"{representation} | {learner_name} | {scope} | "
                    f"AUC={rows[-1]['roc_auc']:.3f} | PR-AUC={rows[-1]['pr_auc']:.3f}"
                )
        pd.DataFrame(rows).to_csv(OUTPUT / "results.csv", index=False)
        predictions.to_parquet(OUTPUT / "predictions.parquet", index=False)
        status(f"Checkpointed results after {representation}")
        del source, candidate, features
        gc.collect()

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT / "results.csv", index=False)
    predictions.to_parquet(OUTPUT / "predictions.parquet", index=False)
    manifest = {
        "models": MODELS,
        "learners": [
            "Logistic Regression",
            "XGBoost",
        ],
        "pair_features": "absolute difference, elementwise product, cosine similarity",
        "pooling": "attention-mask mean pooling",
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "seed": SEED,
        "bootstrap_iterations": BOOTSTRAPS,
        "device": str(device),
        "timing": timing,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    status("All requested representation ablations completed")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
