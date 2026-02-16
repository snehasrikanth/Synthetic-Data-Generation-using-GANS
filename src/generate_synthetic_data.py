
from __future__ import annotations

import argparse
import json
from typing import Dict

import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata # Describes what each column in your data means
from sdv.single_table import CTGANSynthesizer #The main AI that creates fake data

from config import (
    ALL_COLUMNS,
    BATCH_SIZE,
    CATEGORICAL_COLUMNS,
    DATA_PATH,
    DISCRIMINATOR_DIM,
    EMBEDDING_DIM,
    GAN_EPOCHS,
    GENERATOR_DIM,
    IDENTIFIER_COLUMNS,
    INTEGER_COLUMNS,
    MODEL_DIR,
    MODEL_STATE_PATH,
    NUMERIC_COLUMNS,
    OUTPUT_DIR,
    PAC,
    SEED,
    SYNTHETIC_OUTPUT_PATH,
    SYNTHETIC_SAMPLE_SIZE,
)

# Deterministic sampling through CTGAN random_state configuration
np.random.seed(SEED)

#prepare output directories
def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

#tell what each columns means Eg: categorical, numerical, identifier
def build_metadata(real_df: pd.DataFrame) -> SingleTableMetadata:
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_df)

    for column in NUMERIC_COLUMNS:
        metadata.update_column(column_name=column, sdtype="numerical")

    for column in CATEGORICAL_COLUMNS:
        metadata.update_column(column_name=column, sdtype="categorical")

    for column in IDENTIFIER_COLUMNS:
        metadata.update_column(column_name=column, sdtype="id")

    return metadata

#build the CTGAN model with hyperparameters from config.py
def make_model(metadata: SingleTableMetadata) -> CTGANSynthesizer:
    return CTGANSynthesizer(
        metadata,
        epochs=GAN_EPOCHS,
        batch_size=BATCH_SIZE,
        generator_dim=GENERATOR_DIM,
        discriminator_dim=DISCRIMINATOR_DIM,
        embedding_dim=EMBEDDING_DIM,
        pac=PAC,
        verbose=True,
    )


def _identifier_prefix(value: str) -> str:
    prefix_chars = [ch for ch in value if not ch.isdigit()]
    return "".join(prefix_chars) or "SYN"

#fix wierd values in the generated data...like kids=3.5 or age=150
def postprocess_samples(samples: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:
    ordered = samples.reindex(columns=ALL_COLUMNS, fill_value=np.nan)

    bounds: Dict[str, Dict[str, float]] = {}
    for column in NUMERIC_COLUMNS:
        ordered[column] = pd.to_numeric(ordered[column], errors="coerce")
        bounds[column] = {
            "min": float(real_df[column].min()),
            "max": float(real_df[column].max()),
            "median": float(real_df[column].median()),
        }

    for column in NUMERIC_COLUMNS:
        lower = bounds[column]["min"]
        upper = bounds[column]["max"]
        median = bounds[column]["median"]
        ordered[column] = (
            ordered[column]
            .fillna(median)
            .clip(lower=lower, upper=upper)
        )

    for column in INTEGER_COLUMNS:
        ordered[column] = pd.to_numeric(ordered[column], errors="coerce")
        lower = bounds[column]["min"]
        upper = bounds[column]["max"]
        median = bounds[column]["median"]
        ordered[column] = (
            ordered[column]
            .fillna(median)
            .round()
            .clip(lower=lower, upper=upper)
            .astype(int)
        )

    for column in CATEGORICAL_COLUMNS:
        fallback = real_df[column].mode(dropna=True)
        fallback_value = fallback.iloc[0] if not fallback.empty else "unknown"
        ordered[column] = ordered[column].fillna(fallback_value)

    if IDENTIFIER_COLUMNS:
        base_col = IDENTIFIER_COLUMNS[0]
        example_value = str(real_df[base_col].dropna().iloc[0]) if not real_df[base_col].dropna().empty else "SYN"
        prefix = _identifier_prefix(example_value)
        ordered[base_col] = [f"{prefix}{i:05d}" for i in range(len(ordered))]

    return ordered


def persist_artifacts(model: CTGANSynthesizer, synthetic_df: pd.DataFrame, metadata: SingleTableMetadata) -> None:
    SYNTHETIC_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    synthetic_df.to_csv(SYNTHETIC_OUTPUT_PATH, index=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_STATE_PATH))

    metadata_path = OUTPUT_DIR / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata.to_dict(), fp, indent=2)


def main(sample_count: int = SYNTHETIC_SAMPLE_SIZE) -> None:
    ensure_dirs()
    real_df = pd.read_csv(DATA_PATH)
    metadata = build_metadata(real_df)

    model = make_model(metadata)
    model.fit(real_df) #train the CTGAN model

    synthetic_samples = model.sample(num_rows=sample_count) #create fake rows
    synthetic_df = postprocess_samples(synthetic_samples, real_df)

    persist_artifacts(model, synthetic_df, metadata)
    print(f"Synthetic dataset persisted to {SYNTHETIC_OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic CFO dataset with CTGAN.")
    parser.add_argument(
        "--samples",
        type=int,
        default=SYNTHETIC_SAMPLE_SIZE,
        help="Number of synthetic records to sample.",
    )
    args = parser.parse_args()
    main(args.samples)
