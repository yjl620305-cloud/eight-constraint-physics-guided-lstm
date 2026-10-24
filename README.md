# Eight-Constraint Physics-Guided LSTM for Dam Deformation Forecasting

This repository contains a configurable implementation of an eight-constraint physics-guided LSTM for one-step-ahead dam deformation forecasting. It is intended to accompany the manuscript and does **not** contain the original dam monitoring dataset.

> **Validation status:** This implementation has been run on the authors' authorized confidential monitoring dataset. The authors confirmed that the implementation, constraint definitions, experimental settings, and reported results are consistent with the final manuscript. The monitoring data are not distributed because of confidentiality restrictions.

## 1. Repository contents

- `eight_constraint_lstm.py`: model, data loading, time-ordered preprocessing, scaling, eight soft-constraint losses, training, evaluation, and CSV/JSON result export.
- `requirements.txt`: Python package dependencies.
- `.gitignore`: excludes local datasets, model checkpoints, and generated outputs from version control.
- `DATA_FORMAT.md`: required local data layout and column conventions.

## 2. Data privacy

The raw dam monitoring data are not included because they are subject to confidentiality restrictions. Store authorized data locally and do not commit the CSV/XLSX files, exported raw measurements, or other restricted engineering information to this repository.

## 3. Environment

Python 3.10 or newer is recommended. Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

Run the synthetic architecture/loss check (no dam data are read):

```bash
python eight_constraint_lstm.py --smoke-test
```

## 4. Run training and evaluation

The `--data-dir` argument must point to the **root folder containing both expected subfolders** listed in `DATA_FORMAT.md`.

```bash
python eight_constraint_lstm.py --data-dir "D:/your/local/dam-data-root" --output-dir "outputs/eight_constraint_lstm"
```

Useful options:

```bash
python eight_constraint_lstm.py --help
python eight_constraint_lstm.py --data-dir "D:/your/local/dam-data-root" --epochs 300 --batch-size 32 --sequence-length 14 --seed 42 --device auto
```

The script uses a chronological split: the first four years form the model-development period; the final 20% of that period is reserved for validation; later observations form the test set. Feature and target scalers are fitted only on the earlier training interval. Each one-step forecast uses the preceding 14 aligned observations. For every forecast in a physical-constraint group, the LSTM receives its own full 14-observation lookback.

The script exports:

- `best_model.pt`: model weights and checkpoint metadata.
- `training_history.csv`: training loss components, validation objective, and learning rate by epoch.
- `test_predictions.csv`: test timestamps, actual displacement, predicted displacement, and errors.
- `test_metrics.json`: R², MAE, RMSE, range-normalized MAPE, Kendall's tau, directional accuracy, and predicted-decrease rate.
- `config_used.json`, `scalers.json`, and `feature_columns.json`: configuration and preprocessing metadata.

The test metrics are calculated once from the selected best-validation checkpoint. Repeating evaluation of the same fixed checkpoint is not counted as independent model runs.

## 5. Run comparison models

Run all supplied baselines with the same chronological split and target alignment:

```bash
python run_all_comparisons.py --data-dir "D:/your/local/dam-data-root" --output-dir "comparison_outputs"
```

Run one model at a time with the compatibility entry points, for example:

```bash
python "无约束LSTM.py" --data-dir "D:/your/local/dam-data-root"
python "PINN-LSTM.py" --data-dir "D:/your/local/dam-data-root"
python "Physics Feature LSTM.py" --data-dir "D:/your/local/dam-data-root"
python "PhysicsGuided Transformer.py" --data-dir "D:/your/local/dam-data-root"
```

Available comparison model names are `lstm`, `pinn_lstm`, `physics_feature_lstm`, `physics_guided_transformer`, `hst`, `arima`, `bilstm`, and `transformer`.

To exercise every comparison-model data flow with generated data, without reading the confidential dataset, run:

```bash
python tests/synthetic_smoke_test.py
```

This is an implementation smoke test only; its metrics are not manuscript results.

## 6. Eight soft constraints

The code implements normalized, differentiable sequence-level penalties:

1. **C1 — temporal monotonicity:** discourages negative predicted deformation rates under the manuscript's positive-displacement sign convention.
2. **C2 — water-level response:** encourages the predicted displacement trend to align with informative water-level changes.
3. **C3 — delayed thermal response:** uses changes in the causal seven-observation moving-average temperature feature as a low-fidelity thermal-response cue.
4. **C4 — acceleration control:** penalizes the magnitude of second-order deformation change above the configured threshold.
5. **C5 — high-water operating stage:** identifies high-water conditions using a training-data quantile, not fixed calendar flood dates, and limits excessive positive growth.
6. **C6 — normal/low-water rate control:** constrains deformation-rate magnitude outside the high-water regime.
7. **C7 — low-fidelity elastic-response residual:** penalizes the residual of a regularized local linear relation between predicted deformation increments and temperature/water-level increments.
8. **C8 — delayed water-level/seepage coupling residual:** penalizes the residual of a regularized local relation between two-step predicted deformation changes and two-step water-level changes.

The defaults are configurable in the `Config` dataclass. The values used in the validated experiment are 0.02 mm/day² for C4, 0.30 mm/day for the high-water rate threshold, and 0.20 mm/day for the normal/low-water rate threshold; the high-water level threshold is the 80th percentile of the training portion. These settings, sign conventions, and the C7/C8 definitions correspond to the final manuscript, but they must not be interpreted as universal physical laws.

## 7. Important methodological details

- Temporal rates and acceleration use discrete finite differences between consecutive forecast outputs; C7 and C8 use differentiable regression residuals.
- The temperature cleaning bounds, feature alignment tolerance, high-water quantile, and physical thresholds are explicit settings corresponding to the approved experimental protocol.
- Water level is aligned backward in time to avoid using a future water-level observation as an input. Displacement is matched to the nearest measurement within the specified tolerance, reflecting the supplied file layout. Inspect the timestamp alignment before relying on the results.
- A 14-observation lookback corresponds to 14 calendar days only when the aligned data are approximately daily. The script warns when the median sampling interval is not close to one day.
- The released preprocessing, loss calculation, forecasting window, and test-evaluation pipeline was run on the authorized dataset and validated against the final manuscript.

## 8. License

No license is granted with this repository. The source code is publicly available for review and reproducibility assessment; reuse or redistribution requires permission from the copyright holders unless otherwise permitted by applicable law.

