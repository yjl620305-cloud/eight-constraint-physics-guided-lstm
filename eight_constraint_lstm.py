"""Eight-constraint physics-guided LSTM for dam deformation forecasting.

This script provides the validated, configurable implementation used for the final
manuscript. It intentionally does not bundle or expose the confidential dam monitoring
data. See README.md for reproduction instructions.

Expected data layout under --data-dir:
  察汗乌苏数据/data/Ts01.csv
  察汗乌苏数据/data/MD03-C2.csv
  监测指标公式及示例测点参数统计/库水位.xlsx

The CSV files must contain columns named `time` and `data`. The Excel file must
contain the water-level timestamp in column B and level in column G (0-based
indices 1 and 6), matching the layout assumed by the supplied code.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from scipy.stats import kendalltau
from torch.utils.data import DataLoader, Dataset


FEATURE_COLUMNS = [
    "timestamp",
    "temperature",
    "water_level",
    "temperature_lag7",
    "water_level_lag7",
    "year_sin",
    "year_cos",
    "month_sin",
    "month_cos",
]
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_COLUMNS)}


@dataclass
class Config:
    start_date: str = "2012-08-13"
    end_date: str = "2017-08-13"
    training_years: int = 4
    validation_fraction_of_training_period: float = 0.20
    sequence_length: int = 14
    rolling_window: int = 7
    max_environment_alignment_days: int = 3
    temperature_min_c: float = -10.0
    temperature_max_c: float = 40.0

    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.30
    head_hidden_size: int = 64

    batch_size: int = 32
    epochs: int = 300
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    random_seed: int = 42
    num_workers: int = 0

    # Physical thresholds correspond to the final manuscript and the validated
    # dam-specific calibration. They remain configurable for other datasets.
    acceleration_threshold_mm_day2: float = 0.02
    high_water_rate_threshold_mm_day: float = 0.30
    normal_water_rate_threshold_mm_day: float = 0.20
    high_water_quantile: float = 0.80
    min_water_change_m: float = 0.01
    min_temperature_change_c: float = 0.05
    thermal_feature_index: int = 3  # temperature_lag7
    water_feature_index: int = 2

    physics_weight: float = 1.0
    ridge_epsilon: float = 1e-3
    rate_normalizer_mm_day: float = 0.20

    # Scheduler follows the supplied implementation's 150-epoch transition.
    cosine_epochs: int = 150
    cosine_min_lr: float = 1e-5
    plateau_factor: float = 0.5
    plateau_patience: int = 10
    plateau_cooldown: int = 5
    plateau_min_lr: float = 1e-6


# ---------------------------------------------------------------------------
# Reproducibility and data loading
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic mode may reduce performance but improves repeatability.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_datetime(values: pd.Series) -> pd.Series:
    """Parse common English/Chinese date strings without forcing one format."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, errors="coerce")
    normalized = values.astype(str).str.strip()
    normalized = normalized.str.replace("年", "/", regex=False)
    normalized = normalized.str.replace("月", "/", regex=False)
    normalized = normalized.str.replace("日", " ", regex=False)
    normalized = normalized.str.replace("-", "/", regex=False)
    try:
        return pd.to_datetime(normalized, format="mixed", errors="coerce")
    except (TypeError, ValueError):
        return pd.to_datetime(normalized, errors="coerce")


def read_series_csv(path: Path, value_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file was not found: {path}")
    frame = pd.read_csv(path)
    frame.columns = [str(c).strip() for c in frame.columns]
    if not {"time", "data"}.issubset(frame.columns):
        raise ValueError(
            f"{path.name} must contain columns named 'time' and 'data'; "
            f"found columns: {frame.columns.tolist()}"
        )
    frame = frame[["time", "data"]].rename(columns={"data": value_name})
    frame["time"] = parse_datetime(frame["time"])
    frame[value_name] = pd.to_numeric(frame[value_name], errors="coerce")
    frame = frame.dropna(subset=["time", value_name]).sort_values("time")
    frame = frame.drop_duplicates(subset="time", keep="last").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No valid records could be read from {path}")
    return frame


def load_raw_data(data_dir: Path, cfg: Config) -> pd.DataFrame:
    root = data_dir
    temp_path = root / "察汗乌苏数据" / "data" / "Ts01.csv"
    disp_path = root / "察汗乌苏数据" / "data" / "MD03-C2.csv"
    water_path = root / "监测指标公式及示例测点参数统计" / "库水位.xlsx"

    temperature = read_series_csv(temp_path, "temperature")
    displacement = read_series_csv(disp_path, "displacement")
    if not water_path.exists():
        raise FileNotFoundError(f"Required input file was not found: {water_path}")

    water_raw = pd.read_excel(water_path, usecols=[1, 6], engine="openpyxl")
    if water_raw.shape[1] != 2:
        raise ValueError("库水位.xlsx must contain the expected timestamp and level columns (B and G).")
    water_raw.columns = ["time", "water_level"]
    water_raw["time"] = parse_datetime(water_raw["time"])
    water_raw["water_level"] = pd.to_numeric(water_raw["water_level"], errors="coerce")
    water = water_raw.dropna(subset=["time", "water_level"]).sort_values("time")
    # Match the daily water-level alignment used in the supplied implementation.
    water["time"] = water["time"].dt.floor("D")
    water = water.groupby("time", as_index=False)["water_level"].last()
    water = water.drop_duplicates(subset="time", keep="last").reset_index(drop=True)
    if water.empty:
        raise ValueError("No valid water-level records were read from 库水位.xlsx.")

    # Align environmental variables backward in time to avoid using a future
    # water-level observation as a feature. Displacement remains the target and
    # is matched to the nearest measurement, matching the original data layout.
    merged = pd.merge_asof(
        temperature.sort_values("time"),
        water.sort_values("time"),
        on="time",
        direction="backward",
        tolerance=pd.Timedelta(days=cfg.max_environment_alignment_days),
    )
    merged = pd.merge_asof(
        merged.sort_values("time"),
        displacement.sort_values("time"),
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta(days=cfg.max_environment_alignment_days),
    )

    start = pd.Timestamp(cfg.start_date)
    end = pd.Timestamp(cfg.end_date)
    merged = merged[(merged["time"] >= start) & (merged["time"] <= end)].copy()
    merged = merged.drop_duplicates(subset="time", keep="last").sort_values("time")
    merged = merged.dropna(subset=["time", "temperature", "water_level", "displacement"])
    if len(merged) <= cfg.sequence_length + 10:
        raise ValueError(
            f"Only {len(merged)} aligned rows remain between {start.date()} and {end.date()}; "
            "check the input files, timestamps, and alignment tolerance."
        )

    temp_bad = (merged["temperature"] < cfg.temperature_min_c) | (
        merged["temperature"] > cfg.temperature_max_c
    )
    merged.loc[temp_bad, "temperature"] = np.nan
    # Forward-only filling prevents a missing feature at time t from using a
    # later measurement. Initial rows with no prior value are dropped.
    merged["temperature"] = merged["temperature"].ffill()
    merged["water_level"] = merged["water_level"].ffill()
    merged = merged.dropna(subset=["temperature", "water_level", "displacement"])

    # Features are all causal: rolling values use the current/past observations.
    merged["timestamp"] = (merged["time"] - merged["time"].min()).dt.total_seconds() / 86400.0
    merged["temperature_lag7"] = merged["temperature"].rolling(
        cfg.rolling_window, min_periods=1
    ).mean()
    merged["water_level_lag7"] = merged["water_level"].rolling(
        cfg.rolling_window, min_periods=1
    ).mean()

    day_of_year = merged["time"].dt.dayofyear.astype(float)
    day_of_month = merged["time"].dt.day.astype(float)
    merged["year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    merged["year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    merged["month_sin"] = np.sin(2 * np.pi * day_of_month / 30.4375)
    merged["month_cos"] = np.cos(2 * np.pi * day_of_month / 30.4375)

    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(
        subset=FEATURE_COLUMNS + ["displacement"]
    )
    merged = merged.sort_values("time").reset_index(drop=True)

    # A 14-row window is a 14-day window only if the aligned observations are
    # daily. Warn rather than silently resample or fabricate displacement labels.
    gaps = merged["time"].diff().dt.total_seconds().dropna() / 86400.0
    median_gap = float(gaps.median()) if len(gaps) else float("nan")
    if np.isfinite(median_gap) and abs(median_gap - 1.0) > 0.25:
        warnings.warn(
            f"The median interval between aligned rows is {median_gap:.3f} days, not about 1 day. "
            "The model uses 14 observations, which may not mean 14 calendar days. Check sampling cadence.",
            RuntimeWarning,
        )
    return merged


def prepare_scaled_data(df: pd.DataFrame, cfg: Config):
    start = pd.Timestamp(cfg.start_date)
    train_end = start + pd.DateOffset(years=cfg.training_years)
    val_start = start + pd.to_timedelta(
        cfg.training_years * (1.0 - cfg.validation_fraction_of_training_period) * 365.25,
        unit="D",
    )
    fit_mask = df["time"] < val_start
    if int(fit_mask.sum()) < cfg.sequence_length + 5:
        raise ValueError("Too few training-period rows to fit scalers and create sequence windows.")

    feature_scalers: Dict[str, RobustScaler] = {}
    scaled_features = np.zeros((len(df), len(FEATURE_COLUMNS)), dtype=np.float32)
    for j, name in enumerate(FEATURE_COLUMNS):
        scaler = RobustScaler()
        scaler.fit(df.loc[fit_mask, [name]])
        feature_scalers[name] = scaler
        scaled_features[:, j] = scaler.transform(df[[name]]).astype(np.float32).ravel()

    target_scaler = RobustScaler().fit(df.loc[fit_mask, ["displacement"]])
    target_scaled = target_scaler.transform(df[["displacement"]]).astype(np.float32).ravel()
    high_water_threshold = float(
        df.loc[fit_mask, "water_level"].quantile(cfg.high_water_quantile)
    )

    splits = {
        "train": (df["time"] >= start) & (df["time"] < val_start),
        "validation": (df["time"] >= val_start) & (df["time"] < train_end),
        "test": (df["time"] >= train_end) & (df["time"] <= pd.Timestamp(cfg.end_date)),
    }
    metadata = {
        "start_date": start,
        "validation_start": val_start,
        "test_start": train_end,
        "end_date": pd.Timestamp(cfg.end_date),
        "high_water_threshold": high_water_threshold,
        "fit_rows": int(fit_mask.sum()),
    }
    return scaled_features, target_scaled, feature_scalers, target_scaler, metadata, splits


class ForecastGroupDataset(Dataset):
    """Groups of consecutive one-step forecasts with a full lookback per target.

    Each target row t receives its own input window [t-sequence_length, t), so
    every forecast—not just the last forecast in a window—uses the same number
    of historical observations. Consecutive target forecasts within each group
    allow the eight soft constraints to operate on an aligned prediction series.
    The final group may overlap earlier targets to retain the split's tail.
    """
    def __init__(self, x, y, times, indices, sequence_length: int):
        self.x = x
        self.y = y
        self.times = pd.to_datetime(times).to_numpy(dtype="datetime64[ns]")
        self.sequence_length = sequence_length
        valid = np.asarray(indices, dtype=np.int64)
        valid = valid[valid >= sequence_length]
        valid = np.unique(valid)
        if len(valid) == 0:
            raise ValueError("This split has no eligible target rows after applying the lookback window.")
        group_size = min(sequence_length, len(valid))
        starts = list(range(0, len(valid) - group_size + 1, group_size))
        last_start = len(valid) - group_size
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
        self.groups = [valid[s:s + group_size] for s in starts]

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        target_indices = self.groups[idx]
        windows = np.stack([
            self.x[t - self.sequence_length:t] for t in target_indices
        ]).astype(np.float32)
        targets = self.y[target_indices].astype(np.float32).reshape(-1, 1)
        target_times_ns = self.times[target_indices].astype("datetime64[ns]").astype("int64")
        dt_days = np.diff(target_times_ns).astype(np.float64) / 86_400_000_000_000.0
        dt_days = np.maximum(dt_days, 1e-3).astype(np.float32)
        return (
            torch.from_numpy(windows),
            torch.from_numpy(targets),
            torch.from_numpy(dt_days),
            torch.from_numpy(target_indices.astype(np.int64)),
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class PhysicsLSTM(nn.Module):
    """LSTM that forecasts the next displacement from each fixed-length window."""
    def __init__(
        self,
        input_size: int = len(FEATURE_COLUMNS),
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.30,
        head_hidden_size: int = 64,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(head_hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Accept [B, group_size, lookback, features], return [B, group_size, 1]."""
        if x.ndim != 4:
            raise ValueError(
                "Expected grouped forecast windows [batch, group, lookback, features], "
                f"got {tuple(x.shape)}"
            )
        batch, group_size, lookback, features = x.shape
        flattened = x.reshape(batch * group_size, lookback, features)
        flattened = self.input_norm(flattened)
        hidden, _ = self.lstm(flattened)
        last_hidden = self.hidden_norm(hidden[:, -1, :])
        forecast = self.head(last_hidden)
        return forecast.reshape(batch, group_size, 1)


# ---------------------------------------------------------------------------
# Differentiable eight-constraint loss
# ---------------------------------------------------------------------------
class EightConstraintPhysicsLoss(nn.Module):
    """Eight normalized, differentiable sequence-level soft constraints.

    This implementation defines discrete physical constraints over the
    predicted displacement sequence, using actual elapsed time and physical units.

    The sign conventions and thresholds correspond to the validated final
    manuscript. All thresholds remain configurable in Config.
    """
    def __init__(
        self,
        feature_scalers: Dict[str, RobustScaler],
        target_scaler: RobustScaler,
        high_water_threshold: float,
        cfg: Config,
    ):
        super().__init__()
        self.cfg = cfg
        self.high_water_threshold = float(high_water_threshold)
        self.register_buffer("target_center", torch.tensor(float(target_scaler.center_[0])))
        self.register_buffer("target_scale", torch.tensor(max(float(target_scaler.scale_[0]), 1e-8)))
        for feature_name in ["water_level", "temperature_lag7"]:
            scaler = feature_scalers[feature_name]
            center = float(scaler.center_[0])
            scale = max(float(scaler.scale_[0]), 1e-8)
            self.register_buffer(f"{feature_name}_center", torch.tensor(center))
            self.register_buffer(f"{feature_name}_scale", torch.tensor(scale))

    @staticmethod
    def _safe_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.numel() == 0 or not bool(mask.any().item()):
            return values.sum() * 0.0
        return values[mask].mean()

    def _inverse_feature(self, values: torch.Tensor, name: str) -> torch.Tensor:
        center = getattr(self, f"{name}_center")
        scale = getattr(self, f"{name}_scale")
        return values * scale + center

    def _ridge_residual_loss(self, response: torch.Tensor, predictors: torch.Tensor) -> torch.Tensor:
        """Batch-wise ridge residual for low-fidelity linear coupling constraints."""
        # response: [B,T], predictors: [B,T,P]
        if response.shape[1] < predictors.shape[-1] + 2:
            return response.sum() * 0.0
        predictor_scale = torch.sqrt(torch.mean(predictors.detach() ** 2, dim=1, keepdim=True)).clamp_min(1e-3)
        predictors_scaled = predictors / predictor_scale
        ones = torch.ones_like(response).unsqueeze(-1)
        design = torch.cat([predictors_scaled, ones], dim=-1)
        xtx = design.transpose(1, 2) @ design
        xty = design.transpose(1, 2) @ response.unsqueeze(-1)
        p = design.shape[-1]
        ridge = torch.eye(p, device=response.device, dtype=response.dtype).unsqueeze(0) * self.cfg.ridge_epsilon
        # Do not penalize the intercept.
        ridge[:, -1, -1] = 0.0
        coeff = torch.linalg.solve(xtx + ridge, xty)
        residual = response.unsqueeze(-1) - design @ coeff
        scale = self.target_scale.to(device=response.device, dtype=response.dtype).clamp_min(1e-3)
        return torch.mean((residual.squeeze(-1) / scale) ** 2)

    def forward(
        self,
        x_norm: torch.Tensor,
        predictions_norm: torch.Tensor,
        dt_days: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # x_norm is [B, group_size, lookback, features]; the final row of each
        # lookback is the environmental context available for that forecast.
        pred_mm = predictions_norm.squeeze(-1) * self.target_scale + self.target_center
        context = x_norm[:, :, -1, :]
        if pred_mm.shape[1] < 3:
            zero = pred_mm.sum() * 0.0
            return zero, {f"C{i}": zero for i in range(1, 9)}

        dt = dt_days.to(dtype=pred_mm.dtype, device=pred_mm.device).clamp_min(1e-3)
        delta_d = pred_mm[:, 1:] - pred_mm[:, :-1]
        rate = delta_d / dt
        rate_norm = max(self.cfg.rate_normalizer_mm_day, 1e-6)

        # C1: temporal monotonicity in the manuscript's positive-displacement convention.
        c1 = F.relu(-rate / rate_norm).mean()

        # Convert the final known environmental context in each forecast window
        # back to physical units. All values are strictly historical inputs.
        water = self._inverse_feature(context[:, :, FEATURE_INDEX["water_level"]], "water_level")
        temp_lag = self._inverse_feature(
            context[:, :, FEATURE_INDEX["temperature_lag7"]], "temperature_lag7"
        )
        delta_water = water[:, 1:] - water[:, :-1]
        delta_temp_lag = temp_lag[:, 1:] - temp_lag[:, :-1]

        # C2: displacement trend should align with changes in the known water-level
        # context when those changes are large enough to be informative.
        water_mask = delta_water.abs() >= self.cfg.min_water_change_m
        c2_values = F.relu(-(torch.sign(delta_water) * rate) / rate_norm)
        c2 = self._safe_masked_mean(c2_values, water_mask)

        # C3: delayed thermal response, approximated using the causal 7-record
        # temperature moving-average feature supplied to the model.
        temp_mask = delta_temp_lag.abs() >= self.cfg.min_temperature_change_c
        c3_values = F.relu(-(torch.sign(delta_temp_lag) * rate) / rate_norm)
        c3 = self._safe_masked_mean(c3_values, temp_mask)

        # C4: acceleration magnitude should remain under its historical threshold.
        mean_dt = ((dt[:, 1:] + dt[:, :-1]) * 0.5).clamp_min(1e-3)
        acceleration = (rate[:, 1:] - rate[:, :-1]) / mean_dt
        accel_threshold = max(self.cfg.acceleration_threshold_mm_day2, 1e-6)
        c4 = F.relu(acceleration.abs() - accel_threshold).mean() / accel_threshold

        # C5/C6: use an operating-stage split, not fixed calendar flood dates.
        # C5 high-water regime: limit excessive positive deformation growth.
        high_water_mask = water[:, 1:] >= self.high_water_threshold
        high_thr = max(self.cfg.high_water_rate_threshold_mm_day, 1e-6)
        c5_values = F.relu(rate - high_thr) / high_thr
        c5 = self._safe_masked_mean(c5_values, high_water_mask)

        # C6 normal/low-water regime: constrain the magnitude of the rate.
        normal_water_mask = ~high_water_mask
        normal_thr = max(self.cfg.normal_water_rate_threshold_mm_day, 1e-6)
        c6_values = F.relu(rate.abs() - normal_thr) / normal_thr
        c6 = self._safe_masked_mean(c6_values, normal_water_mask)

        # C7: low-fidelity elastic-response approximation. The sequence of
        # predicted displacement increments is fitted to temperature and water
        # increments; unexplained residuals are softly penalized.
        delta_temp_norm = context[:, 1:, FEATURE_INDEX["temperature_lag7"]] - context[:, :-1, FEATURE_INDEX["temperature_lag7"]]
        delta_water_norm = context[:, 1:, FEATURE_INDEX["water_level"]] - context[:, :-1, FEATURE_INDEX["water_level"]]
        elastic_predictors = torch.stack([delta_temp_norm, delta_water_norm], dim=-1)
        c7 = self._ridge_residual_loss(delta_d, elastic_predictors)

        # C8: delayed seepage/water-level coupling based on two-step changes.
        if pred_mm.shape[1] > 3:
            delta_d_lag2 = pred_mm[:, 2:] - pred_mm[:, :-2]
            delta_water_norm_lag2 = context[:, 2:, FEATURE_INDEX["water_level"]] - context[:, :-2, FEATURE_INDEX["water_level"]]
            c8 = self._ridge_residual_loss(delta_d_lag2, delta_water_norm_lag2.unsqueeze(-1))
        else:
            c8 = pred_mm.sum() * 0.0

        terms = {f"C{i}": value for i, value in enumerate([c1, c2, c3, c4, c5, c6, c7, c8], start=1)}
        physics_loss = torch.stack(list(terms.values())).mean()
        return physics_loss, terms


# ---------------------------------------------------------------------------
# Training/evaluation
# ---------------------------------------------------------------------------
def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    actual = np.asarray(actual, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    mask = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[mask], predicted[mask]
    if actual.size == 0:
        raise ValueError("No finite test predictions were available.")
    value_range = float(np.max(actual) - np.min(actual))
    nmape = float(np.mean(np.abs(predicted - actual)) / max(value_range, 1e-8) * 100.0)
    tau_result = kendalltau(actual, predicted)
    tau = float(tau_result.statistic) if np.isfinite(tau_result.statistic) else float("nan")
    da = float(np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(predicted))) * 100.0) if actual.size > 1 else float("nan")
    violation_rate = float(np.mean(np.diff(predicted) < 0) * 100.0) if actual.size > 1 else float("nan")
    return {
        "n": int(actual.size),
        "R2": float(r2_score(actual, predicted)) if actual.size > 1 else float("nan"),
        "MAE_mm": float(mean_absolute_error(actual, predicted)),
        "RMSE_mm": float(math.sqrt(mean_squared_error(actual, predicted))),
        "nMAPE_percent_range_normalized": nmape,
        "Kendall_tau": tau,
        "directional_accuracy_percent": da,
        "predicted_decrease_rate_percent": violation_rate,
    }


def batch_objective(
    model: PhysicsLSTM,
    physics_loss_fn: EightConstraintPhysicsLoss,
    x: torch.Tensor,
    y_targets: torch.Tensor,
    dt_days: torch.Tensor,
    cfg: Config,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    predictions = model(x)
    data_loss = F.mse_loss(predictions, y_targets)
    phys_loss, constraint_terms = physics_loss_fn(x, predictions, dt_days)
    objective = data_loss + cfg.physics_weight * phys_loss
    values = {"data_mse": data_loss, "physics_loss": phys_loss, **constraint_terms}
    return objective, values


def evaluate_split(
    model: PhysicsLSTM,
    loader: DataLoader,
    physics_loss_fn: EightConstraintPhysicsLoss,
    device: torch.device,
    cfg: Config,
    target_scaler: RobustScaler,
):
    model.eval()
    losses = []
    all_preds, all_actuals, all_target_indices = [], [], []
    with torch.no_grad():
        for x, y_targets, dt_days, target_idx in loader:
            x, y_targets, dt_days = x.to(device), y_targets.to(device), dt_days.to(device)
            preds = model(x)
            phys_loss, _ = physics_loss_fn(x, preds, dt_days)
            data_loss = F.mse_loss(preds, y_targets)
            objective = data_loss + cfg.physics_weight * phys_loss
            losses.append(float(objective.cpu().item()))
            all_preds.append(preds.cpu().numpy().reshape(-1))
            all_actuals.append(y_targets.cpu().numpy().reshape(-1))
            all_target_indices.append(target_idx.numpy().reshape(-1))
    if not losses:
        raise ValueError("The validation/test split produced no batches.")
    pred_norm = np.concatenate(all_preds)
    true_norm = np.concatenate(all_actuals)
    indices = np.concatenate(all_target_indices).astype(int)
    pred_mm = target_scaler.inverse_transform(pred_norm.reshape(-1, 1)).ravel()
    true_mm = target_scaler.inverse_transform(true_norm.reshape(-1, 1)).ravel()
    order = np.argsort(indices)
    indices, pred_mm, true_mm = indices[order], pred_mm[order], true_mm[order]
    # The final group's overlap is intentional to preserve its tail; deduplicate
    # target indices before computing metrics or writing predictions.
    unique_indices, unique_positions = np.unique(indices, return_index=True)
    return float(np.mean(losses)), pred_mm[unique_positions], true_mm[unique_positions], unique_indices


def scaler_summary(feature_scalers: Dict[str, RobustScaler], target_scaler: RobustScaler):
    result = {name: {"center": float(sc.center_[0]), "scale": float(sc.scale_[0])}
              for name, sc in feature_scalers.items()}
    result["displacement"] = {
        "center": float(target_scaler.center_[0]),
        "scale": float(target_scaler.scale_[0]),
    }
    return result


def train_and_evaluate(data_dir: Path, output_dir: Path, cfg: Config, device_string: str = "auto") -> None:
    set_seed(cfg.random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_raw_data(data_dir, cfg)
    x_np, y_np, feature_scalers, target_scaler, meta, split_masks = prepare_scaled_data(df, cfg)

    if device_string == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_string)
    print(f"Device: {device}")
    print(f"Aligned rows: {len(df)}")
    print(f"Validation starts: {meta['validation_start']}")
    print(f"Test starts: {meta['test_start']}")
    print(f"High-water threshold (training quantile): {meta['high_water_threshold']:.6g}")

    times = df["time"].to_numpy()
    index_sets = {
        split_name: np.flatnonzero(mask.to_numpy())
        for split_name, mask in split_masks.items()
    }
    datasets = {
        split_name: ForecastGroupDataset(x_np, y_np, times, indices, cfg.sequence_length)
        for split_name, indices in index_sets.items()
    }
    train_loader = DataLoader(
        datasets["train"], batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=False
    )
    val_loader = DataLoader(
        datasets["validation"], batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, drop_last=False
    )
    test_loader = DataLoader(
        datasets["test"], batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, drop_last=False
    )

    model = PhysicsLSTM(
        input_size=len(FEATURE_COLUMNS), hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers, dropout=cfg.dropout,
        head_hidden_size=cfg.head_hidden_size,
    ).to(device)
    physics_loss_fn = EightConstraintPhysicsLoss(
        feature_scalers=feature_scalers,
        target_scaler=target_scaler,
        high_water_threshold=meta["high_water_threshold"],
        cfg=cfg,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    cosine_epochs = min(cfg.cosine_epochs, cfg.epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cosine_epochs, 1), eta_min=cfg.cosine_min_lr
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.plateau_factor, patience=cfg.plateau_patience,
        cooldown=cfg.plateau_cooldown, min_lr=cfg.plateau_min_lr,
    )

    with (output_dir / "config_used.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
    with (output_dir / "scalers.json").open("w", encoding="utf-8") as f:
        json.dump(scaler_summary(feature_scalers, target_scaler), f, ensure_ascii=False, indent=2)
    (output_dir / "feature_columns.json").write_text(
        json.dumps(FEATURE_COLUMNS, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    best_val = float("inf")
    best_path = output_dir / "best_model.pt"
    history: List[dict] = []
    for epoch in range(cfg.epochs):
        model.train()
        train_values: Dict[str, List[float]] = {}
        for x, y_targets, dt_days, _ in train_loader:
            x, y_targets, dt_days = x.to(device), y_targets.to(device), dt_days.to(device)
            optimizer.zero_grad(set_to_none=True)
            objective, values = batch_objective(model, physics_loss_fn, x, y_targets, dt_days, cfg)
            if not torch.isfinite(objective):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}; inspect data and thresholds.")
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
            optimizer.step()
            for key, value in values.items():
                train_values.setdefault(key, []).append(float(value.detach().cpu().item()))

        train_mean = {key: float(np.mean(vals)) for key, vals in train_values.items()}
        val_objective, _, _, _ = evaluate_split(
            model, val_loader, physics_loss_fn, device, cfg, target_scaler
        )
        if epoch < cosine_epochs:
            cosine_scheduler.step()
        else:
            plateau_scheduler.step(val_objective)
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch + 1,
            **{f"train_{key}": val for key, val in train_mean.items()},
            "validation_objective": val_objective,
            "learning_rate": current_lr,
        }
        history.append(row)
        if val_objective < best_val:
            best_val = val_objective
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "feature_columns": FEATURE_COLUMNS,
                "best_validation_objective": best_val,
                "epoch": epoch + 1,
            }, best_path)
        if (epoch + 1) == 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:03d}/{cfg.epochs} | "
                f"train forecast MSE={train_mean.get('data_mse', float('nan')):.5f} | "
                f"physics={train_mean.get('physics_loss', float('nan')):.5f} | "
                f"val={val_objective:.5f} | lr={current_lr:.2e}"
            )
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    # Reload the best validation checkpoint before the final test evaluation.
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_objective, preds, actuals, target_indices = evaluate_split(
        model, test_loader, physics_loss_fn, device, cfg, target_scaler
    )
    metrics = calculate_metrics(actuals, preds)
    metrics["test_objective_normalized"] = test_objective
    metrics["best_validation_objective_normalized"] = float(best_val)
    metrics["best_epoch"] = int(checkpoint["epoch"])

    results = pd.DataFrame({
        "time": pd.to_datetime(df.iloc[target_indices]["time"].to_numpy()),
        "actual_displacement_mm": actuals,
        "predicted_displacement_mm": preds,
        "error_mm": preds - actuals,
    }).sort_values("time")
    results.to_csv(output_dir / "test_predictions.csv", index=False)
    # Convert NaN/Inf metric values to JSON null to keep the file standards-compliant.
    safe_metrics = {
        key: (value if not isinstance(value, float) or np.isfinite(value) else None)
        for key, value in metrics.items()
    }
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(safe_metrics, f, ensure_ascii=False, indent=2, allow_nan=False)

    print("\nFinal test metrics (single evaluation of the selected checkpoint):")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"\nSaved best model: {best_path}")
    print(f"Saved predictions: {output_dir / 'test_predictions.csv'}")
    print(f"Saved metrics: {output_dir / 'test_metrics.json'}")


def run_synthetic_smoke_test() -> None:
    """Quick architecture/loss/gradient test; it does not use dam data."""
    cfg = Config(sequence_length=14)
    rng = np.random.default_rng(1)
    scalers = {
        name: RobustScaler().fit(rng.normal(size=(100, 1)))
        for name in FEATURE_COLUMNS
    }
    target_scaler = RobustScaler().fit(np.random.default_rng(2).normal(size=(100, 1)))
    model = PhysicsLSTM()
    loss_fn = EightConstraintPhysicsLoss(scalers, target_scaler, 10.0, cfg)
    # 4 grouped samples; each contains 14 individual 14-observation lookbacks.
    x = torch.randn(4, cfg.sequence_length, cfg.sequence_length, len(FEATURE_COLUMNS))
    y = torch.randn(4, cfg.sequence_length, 1)
    dt = torch.ones(4, cfg.sequence_length - 1)
    pred = model(x)
    physics, pieces = loss_fn(x, pred, dt)
    objective = F.mse_loss(pred, y) + cfg.physics_weight * physics
    objective.backward()
    assert pred.shape == (4, cfg.sequence_length, 1)
    assert torch.isfinite(objective)
    assert all(torch.isfinite(v) for v in pieces.values())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    print("Synthetic smoke test passed: grouped full-lookback forecasts, all eight losses, and model gradients are finite.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Root folder containing the dam input-data subfolders")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eight_constraint_lstm"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--smoke-test", action="store_true", help="Run only a synthetic test without loading dam data")
    args = parser.parse_args()

    if args.smoke_test:
        run_synthetic_smoke_test()
        return
    cfg = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        random_seed=args.seed,
    )
    train_and_evaluate(args.data_dir, args.output_dir, cfg, args.device)


if __name__ == "__main__":
    main()

