# Results package

Compressed CSV files with a header row. Read floats exactly with
`pandas.read_csv(path, float_precision="round_trip")`. A run is identified by `model`,
`n_train` (training size N), `fold`, and `seed`.

| Directory | Configuration | Protocol | Runs |
|---|---|---|---|
| `primary/` | `configs/experiment.yaml` | blocked | 2040 |
| `blocked_unit/` | `configs/blocked_unit.yaml` | blocked | 720 |
| `random/` | `configs/random.yaml` | random cross-validation | 2760 |

`manifest.json`: per batch, the configuration, protocol, configuration hash, dataset SHA-256,
and number of runs; the SHA-256 and size of every file.

## `runs.csv.gz`: one row per run

| Column | Content |
|---|---|
| `model`, `kind` | Model name; classical, deep, mlp_pm, or qnn |
| `n_train`, `fold`, `seed` | Run identifier |
| `readout` | Target range the model was trained on: `[-1, 1]` or `[0, 1]` |
| `val_*`, `test_*` | `r2`, `rmse`, `mae`, `bias` (mean of prediction minus actual), `error_std`, `n_negative`, `frac_negative` on the validation block and the test fold (kW) |
| `test_clip_rmse`, `test_clip_r2`, `test_clip_mae` | Test metrics with predictions clipped to [0, maximum training power] |
| `trainable_params` | Trainable parameter count |
| `tree_nodes`, `tree_leaves` | DTR, XGBoost, LightGBM |
| `support_vectors` | SVR |
| `stored_training_samples` | kNN |
| `nit`, `nfev`, `scipy_message`, `stopped_before_maxiter`, `cache_near_hits`, `best_is_initial` | L-BFGS-B summary (QNN, MLP-PM) |
| `time_per_evaluation` | Optimizer time per objective evaluation, s (QNN, MLP-PM) |
| `total_training_time` | s (QNN, MLP-PM, deep models) |
| `node_busy_mean`, `node_busy_min` | Busy workers on the compute node during the run |
| `hyperparameters` | Hyperparameters of the run (JSON) |
| `config_hash`, `dataset_sha256` | SHA-256 of the run's configuration and of the dataset |

## Other files

| File | Columns |
|---|---|
| `errors_test.csv.gz` | run identifier, `row` (0-based row in the dataset), `error_kw` (prediction minus actual, kW), for every test sample |
| `pooled_seed.csv.gz` | `model`, `n_train`, `seed`, `r2`, `rmse`, `mae`, `bias` of the pooled out-of-fold predictions (six test folds, every row once) |
| `curves_iter.csv.gz` | run identifier, `iteration` (0 = initial weights), `train_mse`, `val_mse` (scaled target), `padded` (after an early stop, last value repeated); QNN only |
| `curves_eval.csv.gz` | run identifier, `evaluation`, `train_mse` (scaled target); QNN only |
| `tuning.csv.gz` | `batch`, `model`, `n_train`, `fold`, `params` (JSON), `best_val_rmse_kw`, `trials` |
