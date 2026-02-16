"""Configuration for the synthetic data GAN workflow."""
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "dataset.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
SYNTHETIC_OUTPUT_PATH = OUTPUT_DIR / "synthetic_dataset.csv"
MODEL_DIR = OUTPUT_DIR / "models"
MODEL_STATE_PATH = MODEL_DIR / "ctgan.pkl"
QUALITY_REPORT_PATH = OUTPUT_DIR / "quality_report.json"

NUMERIC_COLUMNS = [
    "user-age",
    "user-income",
    "user-savings",
    "user-properties",
    "user-dependents",
    "user-pension",
    "product-term",
    "year",
    "month",
]

INTEGER_COLUMNS = [
    "user-age",
    "user-properties",
    "user-dependents",
    "product-term",
    "year",
    "month",
]

CATEGORICAL_COLUMNS = [
    "user-gender",
    "user-nationality",
    "user-knowledge",
    "user-loyalty",
    "user-loan",
    "user-riskAversion",
    "user-marital",
    "product-type",
    "product-risk",
    "product-yield",
]

IDENTIFIER_COLUMNS = ["transaction-id", "user-id"]

ALL_COLUMNS = [
    "user-id",
    "user-age",
    "user-gender",
    "user-nationality",
    "user-knowledge",
    "user-loyalty",
    "user-loan",
    "user-income",
    "user-savings",
    "user-properties",
    "user-riskAversion",
    "user-marital",
    "user-dependents",
    "user-pension",
    "product-type",
    "product-risk",
    "product-term",
    "product-yield",
    "transaction-id",
    "year",
    "month",
]

GAN_EPOCHS = 300
BATCH_SIZE = 500
GENERATOR_DIM = (256, 256, 256)
DISCRIMINATOR_DIM = (256, 256, 256)
EMBEDDING_DIM = 128
PAC = 10
SYNTHETIC_SAMPLE_SIZE = 11000
SEED = 42
