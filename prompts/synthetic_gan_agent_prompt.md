# Synthetic CFO GAN Agent Instructions

You are GitHub Copilot working in VS Code agent mode. Build an end-to-end workflow that trains a GAN-based model to generate realistic CFO investment records and evaluates the synthetic data quality.

## Dataset
- Source CSV: `dataset.csv` (≈11k rows) located at the workspace root.
- Columns:
  - Numeric: `user-age`, `user-income`, `user-savings`, `user-properties`, `user-dependents`, `user-pension`, `product-term`, `year`, `month`
  - Categorical: `user-id`, `user-gender`, `user-nationality`, `user-knowledge`, `user-loyalty`, `user-loan`, `user-riskAversion`, `user-marital`, `product-type`, `product-risk`, `product-yield`
  - Identifier: `transaction-id`

## Objectives
1. Train a CTGAN (GAN for tabular data) model on the real dataset.
2. Sample a synthetic dataset with at least 11k rows.
3. Save model artifacts, generated samples, and metadata under `artifacts/`.
4. Run a quality evaluation script and store the report as JSON.

## Required Project Structure
```
src/
  config.py                 # Provides paths, column groupings, and GAN hyperparameters
  generate_synthetic_data.py# Trains CTGAN and writes synthetic samples to artifacts/
  evaluate_synthetic_quality.py # Compares real vs synthetic data and produces metrics
artifacts/
  synthetic_dataset.csv
  models/ctgan.pkl
  metadata.json
  quality_report.json
prompts/
  synthetic_gan_agent_prompt.md  # (this file)
```

## Tasks for the Agent
1. **Environment**
   - Create a Python 3.10+ virtual environment.
   - Install dependencies: `pandas`, `numpy`, `scikit-learn`, `sdv>=1.7`, `pyaml` (for optional debugging).

2. **Model Training**
   - Execute `python src/generate_synthetic_data.py --samples 11000`.
   - Confirm outputs: `artifacts/synthetic_dataset.csv`, `artifacts/models/ctgan.pkl`, `artifacts/metadata.json`.

3. **Quality Evaluation**
   - Run `python src/evaluate_synthetic_quality.py`.
   - Ensure `artifacts/quality_report.json` is created and review console summary.

4. **Verification**
   - Inspect the first few rows of the synthetic dataset (e.g., `head` via pandas) to confirm column presence and reasonable ranges.
   - Note any warnings or convergence issues from CTGAN; rerun with adjusted epochs/batch size if needed.

5. **Handoff Notes**
   - Summarize training duration, key quality metrics, and any manual adjustments in the task log or output panel.

## Constraints & Tips
- Keep random seeds at 42 for reproducibility.
- For integer columns, round and clip values to the observed real-data bounds.
- Ensure `transaction-id` values in the synthetic data remain unique (the generator script already handles this).
- If GPU is unavailable, CTGAN should automatically fall back to CPU; document the device used.
- Avoid committing artifacts; store them only under `artifacts/` for inspection.

Follow these steps sequentially and report progress after each major command.
