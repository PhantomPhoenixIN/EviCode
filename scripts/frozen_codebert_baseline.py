"""Run a frozen CodeBERT source-candidate baseline on the authentic split."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "authentic_only" / "datasets" / "analysis_dataset.parquet"
RAW = ROOT / "datasets" / "Predictions_by_LLMs"
CACHE = ROOT / "outputs" / "authentic_only" / "datasets" / "features.jsonl"
OUTPUT = ROOT / "outputs" / "frozen_codebert_baseline"
MODEL_NAME = "microsoft/codebert-base"
SEED = 42


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def generator(path: Path) -> tuple[str, str]:
    name = path.name.lower()
    if "deepseek" in name:
        return "DeepSeek-Coder", "deepseekcoder"
    if "qwen" in name:
        return "QwenCoder", "qwencoder"
    return "StarCoder", "starcoder"


def load_hashes(wanted: set[str]) -> dict[str, tuple[str, str]]:
    hashes: dict[str, tuple[str, str]] = {}
    with CACHE.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["example_id"] in wanted:
                hashes[record["example_id"]] = (
                    record["source_sha256"],
                    record["candidate_sha256"],
                )
    return hashes


def load_pairs(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    wanted = set(frame["example_id"])
    expected_hashes = load_hashes(wanted)
    records: dict[str, tuple[str, str]] = {}
    for path in sorted(RAW.glob("codenet_single_solution_*_scored*.jsonl")):
        model_name, prefix = generator(path)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                example_id = f"{model_name.lower().replace('-', '')}_{line_number}"
                if example_id not in wanted:
                    continue
                row = json.loads(line)
                source = row.get("source_code") or ""
                candidate = (
                    row.get(f"{prefix}_translation_clean")
                    or row.get("translated_java_code")
                    or row.get(f"{prefix}_translation_raw")
                    or ""
                )
                observed = (
                    hashlib.sha256(source.encode()).hexdigest(),
                    hashlib.sha256(candidate.encode()).hexdigest(),
                )
                if expected_hashes.get(example_id) != observed:
                    raise ValueError(f"Artifact hash mismatch for {example_id}")
                records[example_id] = (source, candidate)
    missing = wanted.difference(records)
    if missing:
        raise ValueError(f"Missing {len(missing)} raw pairs")
    ordered = [records[example_id] for example_id in frame["example_id"]]
    return [pair[0] for pair in ordered], [pair[1] for pair in ordered]


@torch.inference_mode()
def encode(
    texts: list[str],
    tokenizer,
    model,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {name: value.to(device) for name, value in batch.items()}
        output = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1)
        chunks.append(pooled.cpu().numpy().astype(np.float32))
        if start and start % (batch_size * 100) == 0:
            print(f"Encoded {start:,}/{len(texts):,}", flush=True)
    return np.concatenate(chunks)


def encode_unique(
    texts: list[str],
    tokenizer,
    model,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    unique_texts = list(dict.fromkeys(texts))
    print(f"Encoding {len(unique_texts):,} unique texts from {len(texts):,} occurrences", flush=True)
    unique_embeddings = encode(
        unique_texts, tokenizer, model, batch_size, max_length, device
    )
    positions = {text: index for index, text in enumerate(unique_texts)}
    return unique_embeddings[[positions[text] for text in texts]]


def clustered_interval(
    y: np.ndarray,
    probability: np.ndarray,
    groups: np.ndarray,
    iterations: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    unique = np.unique(groups)
    positions = {group: np.flatnonzero(groups == group) for group in unique}
    values: list[float] = []
    for _ in range(iterations):
        sample = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([positions[group] for group in sample])
        if np.unique(y[indices]).size == 2:
            values.append(roc_auc_score(y[indices], probability[indices]))
    return tuple(np.quantile(values, [0.025, 0.975]))


def main() -> None:
    args = arguments()
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pd.read_parquet(DATA)
    if args.max_pairs:
        frame = frame.iloc[: args.max_pairs].copy()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    embedding_path = OUTPUT / f"pair_embeddings_n{len(frame)}_l{args.max_length}.npz"
    source_path = OUTPUT / f"source_embeddings_n{len(frame)}_l{args.max_length}.npy"
    candidate_path = OUTPUT / f"candidate_embeddings_n{len(frame)}_l{args.max_length}.npy"

    start = time.perf_counter()
    if args.resume and embedding_path.exists():
        stored = np.load(embedding_path)
        source_embedding = stored["source"]
        candidate_embedding = stored["candidate"]
    else:
        sources, candidates = load_pairs(frame)
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, local_files_only=not args.allow_download
        )
        model = AutoModel.from_pretrained(
            MODEL_NAME, local_files_only=not args.allow_download
        )
        model.to(device).eval()
        if args.resume and source_path.exists():
            source_embedding = np.load(source_path)
        else:
            source_embedding = encode_unique(
                sources, tokenizer, model, args.batch_size, args.max_length, device
            )
            np.save(source_path, source_embedding)
        if args.resume and candidate_path.exists():
            candidate_embedding = np.load(candidate_path)
        else:
            candidate_embedding = encode_unique(
                candidates, tokenizer, model, args.batch_size, args.max_length, device
            )
            np.save(candidate_path, candidate_embedding)
        np.savez_compressed(
            embedding_path,
            source=source_embedding,
            candidate=candidate_embedding,
        )

    difference = np.abs(source_embedding - candidate_embedding)
    product = source_embedding * candidate_embedding
    cosine = np.sum(source_embedding * candidate_embedding, axis=1, keepdims=True) / (
        np.linalg.norm(source_embedding, axis=1, keepdims=True)
        * np.linalg.norm(candidate_embedding, axis=1, keepdims=True)
    ).clip(min=1e-12)
    features = np.concatenate([difference, product, cosine], axis=1)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_index, test_index = next(
        splitter.split(frame, frame["label"], groups=frame["problem_id"])
    )
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=SEED),
    )
    classifier.fit(features[train_index], frame.iloc[train_index]["label"])
    probability = classifier.predict_proba(features[test_index])[:, 1]
    test = frame.iloc[test_index].copy()
    test["codebert_probability"] = probability

    rows = []
    for scope, selected in {
        "All grades": np.ones(len(test), dtype=bool),
        "Score 2 vs 3": test["quality_score"].isin([2, 3]).to_numpy(),
    }.items():
        y = test.loc[selected, "label"].to_numpy()
        p = test.loc[selected, "codebert_probability"].to_numpy()
        groups = test.loc[selected, "problem_id"].to_numpy()
        low, high = clustered_interval(y, p, groups, args.bootstrap)
        rows.append(
            {
                "model": "Frozen CodeBERT",
                "scope": scope,
                "n": len(y),
                "roc_auc": roc_auc_score(y, p),
                "roc_auc_ci_low": low,
                "roc_auc_ci_high": high,
                "pr_auc": average_precision_score(y, p),
            }
        )

    pd.DataFrame(rows).to_csv(OUTPUT / "results.csv", index=False)
    test[["example_id", "problem_id", "quality_score", "label", "codebert_probability"]].to_parquet(
        OUTPUT / "predictions.parquet", index=False
    )
    joblib.dump(classifier, OUTPUT / "classifier.joblib")
    manifest = {
        "model": MODEL_NAME,
        "pooling": "attention-mask mean pooling",
        "pair_features": "absolute difference, elementwise product, cosine similarity",
        "frozen_encoder": True,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "seed": SEED,
        "pairs": len(frame),
        "embedding_seconds": time.perf_counter() - start,
        "device": str(device),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
