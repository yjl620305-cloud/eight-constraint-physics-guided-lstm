# -*- coding: utf-8 -*-
"""Reproducible comparison models for dam-deformation forecasting.

This module provides a common, time-ordered data pipeline and comparable one-step-ahead
forecast targets for LSTM, PINN-LSTM, Physics Feature LSTM, Physics-Guided Transformer,
HST-style regression, ARIMA, BiLSTM, and vanilla Transformer baselines.

IMPORTANT:
- No raw dam monitoring data are included.
- The validated default forecast window is 14 observations, matching the final manuscript.
- The PINN-LSTM loss is the differentiable, low-fidelity empirical residual used for
  the validated manuscript baseline; it is not a governing PDE.
- Physics-feature models use past-observation-derived features (including lagged
  deformation differences); their information set is therefore not identical to the
  environment-only baselines. This must be stated when interpreting cross-model results.
- ARIMA is an open-loop univariate forecast based on historical displacement, so its
  information set differs from the covariate-driven neural models.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Avoid oversubscribing BLAS/OpenMP threads on Windows/Linux.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import kendalltau
from torch.utils.data import DataLoader, Dataset, TensorDataset

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:  # ARIMA is optional if only neural models are being run.
    ARIMA = None

warnings.filterwarnings("default", category=RuntimeWarning)

# The validated final-manuscript protocol uses a 14-observation input window.
# Override this setting only when running a different experimental protocol.
DEFAULT_SEQ_LEN = 14
DEFAULT_TRAIN_END = "2016-08-13"
DEFAULT_START = "2012-08-13"
DEFAULT_END = "2017-08-13"

BASE_FEATURES = [
    "timestamp_days_norm", "temperature_norm", "water_level_norm",
    "temperature_7d_norm", "water_level_7d_norm",
    "year_sin", "year_cos", "month_sin", "month_cos",
]
PHYS_FEATURES = [
    "phy_mono_norm", "phy_hydro_norm", "phy_thermal_norm", "phy_accel_norm",
    "phy_flood", "phy_rate_norm", "phy_hooke_norm", "phy_seepage_proxy_norm",
]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _first_existing(patterns: Sequence[Path], description: str) -> Path:
    for p in patterns:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot find {description}. Checked:\n" + "\n".join(str(p) for p in patterns)
    )


def _read_two_column_sensor(path: Path, value_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    lower = {str(c).strip().lower(): c for c in frame.columns}
    time_col = lower.get("time")
    value_col = lower.get("data")
    if time_col is None or value_col is None:
        if frame.shape[1] < 2:
            raise ValueError(f"{path.name} must contain at least time and data columns")
        time_col, value_col = frame.columns[:2]
    out = frame[[time_col, value_col]].copy()
    out.columns = ["time", value_name]
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
    out = out.dropna(subset=["time", value_name]).sort_values("time")
    # Match the primary-model loader: preserve the last record at a duplicated timestamp.
    return (out.sort_values("time").drop_duplicates(subset="time", keep="last")
            .reset_index(drop=True))


def _parse_water_times(series: pd.Series) -> pd.Series:
    """Parse Excel date cells, serial dates, and Chinese 年/月/日 strings."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    if pd.api.types.is_numeric_dtype(series):
        # Excel's 1900 date system uses 1899-12-30 as the serial-date origin.
        return pd.to_datetime(series, unit="D", origin="1899-12-30", errors="coerce")
    text = series.astype(str).str.strip()
    text = text.str.replace(r"[年月]", "/", regex=True).str.replace("日", " ", regex=False)
    text = text.str.replace("-", "/", regex=False).str.replace(r"\s+", " ", regex=True)
    parsed = pd.to_datetime(text, errors="coerce")
    return parsed


def load_raw_data(data_dir: str | Path,
                  start_date: str = DEFAULT_START,
                  end_date: str = DEFAULT_END,
                  merge_tolerance_days: int = 3) -> pd.DataFrame:
    """Load raw temperature, displacement, and water-level records.

    Row alignment mirrors the validated primary-model loader: temperature timestamps
    anchor each row; water level is joined backward and displacement is matched to the
    nearest observation within tolerance. This keeps the comparison models on the same
    aligned rows as the main model. Because displacement uses nearest matching, its
    measurement timestamp may differ slightly from the row timestamp, as specified by
    the validated protocol. Missing environmental values are forward-filled only.
    """
    root = Path(data_dir)
    temp_path = _first_existing(
        [root / "察汗乌苏数据" / "data" / "Ts01.csv", root / "data" / "Ts01.csv", root / "Ts01.csv"],
        "temperature file Ts01.csv",
    )
    disp_folders = [root / "察汗乌苏数据" / "data", root / "data", root]
    disp_exact_names = ["MD03-C2.csv", "MD03‑C2.csv", "MD03–C2.csv", "MD03—C2.csv"]
    disp_path = None
    for folder in disp_folders:
        for filename in disp_exact_names:
            candidate = folder / filename
            if candidate.exists():
                disp_path = candidate
                break
        if disp_path is not None:
            break
    if disp_path is None:
        candidates = []
        for folder in disp_folders:
            if folder.exists():
                candidates.extend(sorted(folder.glob("MD03*.csv")))
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) == 1:
            disp_path = candidates[0]
        elif len(candidates) > 1:
            raise FileNotFoundError(
                "Multiple MD03*.csv files were found but no exact MD03-C2 filename was found. "
                "Specify/rename the intended monitoring-point file to MD03-C2.csv. Candidates: "
                + ", ".join(str(p) for p in candidates)
            )
        else:
            raise FileNotFoundError("Cannot find displacement file MD03-C2.csv (or a unique MD03*.csv)")
    level_path = _first_existing(
        [root / "监测指标公式及示例测点参数统计" / "库水位.xlsx", root / "库水位.xlsx"],
        "water-level workbook 库水位.xlsx",
    )

    temperature = _read_two_column_sensor(temp_path, "temperature")
    displacement = _read_two_column_sensor(disp_path, "displacement")

    # Use the same positional columns as the supplied comparison script (2nd and 7th columns).
    raw_level = pd.read_excel(level_path, usecols=[1, 6], engine="openpyxl")
    if raw_level.shape[1] != 2:
        raise ValueError("Water-level workbook must expose the expected time/value columns at positions 2 and 7")
    raw_level.columns = ["time", "water_level"]
    raw_level["time"] = _parse_water_times(raw_level["time"])
    raw_level["water_level"] = pd.to_numeric(raw_level["water_level"], errors="coerce")
    water = raw_level.dropna(subset=["time", "water_level"]).sort_values("time")
    water["time"] = water["time"].dt.floor("D")
    water = (water.sort_values("time").groupby("time", as_index=False)["water_level"].last()
             .sort_values("time").reset_index(drop=True))

    # Match the main PhyLSTM data alignment: temperature timestamps anchor rows;
    # water level uses the latest prior reading; displacement uses nearest matching.
    df = temperature.sort_values("time").copy()
    tolerance = pd.Timedelta(days=merge_tolerance_days)
    df = pd.merge_asof(
        df, water.sort_values("time"), on="time", direction="backward", tolerance=tolerance
    )
    df = pd.merge_asof(
        df, displacement.sort_values("time"), on="time", direction="nearest", tolerance=tolerance
    )

    lo, hi = pd.Timestamp(start_date), pd.Timestamp(end_date)
    df = df[(df["time"] >= lo) & (df["time"] <= hi)].copy()
    df = df.drop_duplicates(subset="time", keep="last").sort_values("time").reset_index(drop=True)
    # Match the primary loader: discard rows without aligned raw measurements first.
    df = df.dropna(subset=["temperature", "water_level", "displacement"]).copy()
    # Exclude implausible air-temperature readings and fill forward only.
    df.loc[(df["temperature"] < -10) | (df["temperature"] > 40), "temperature"] = np.nan
    df[["temperature", "water_level"]] = df[["temperature", "water_level"]].ffill()
    df = df.dropna(subset=["displacement", "temperature", "water_level"]).reset_index(drop=True)
    if len(df) <= DEFAULT_SEQ_LEN + 20:
        raise ValueError(
            f"Only {len(df)} aligned rows remain after cleaning; verify source paths, date formats, "
            "and sensor timestamps."
        )
    if not df["time"].is_monotonic_increasing:
        raise ValueError("Aligned timestamps are not monotonic")
    gaps = df["time"].diff().dt.total_seconds().dropna() / 86400.0
    median_gap = float(gaps.median()) if len(gaps) else float("nan")
    if np.isfinite(median_gap) and abs(median_gap - 1.0) > 0.25:
        warnings.warn(
            f"Median interval between aligned rows is {median_gap:.3f} days, not approximately one day. "
            "The default sequence length counts observations, not necessarily calendar days; "
            "ARIMA also treats each row as one equally spaced step.",
            RuntimeWarning,
        )
    return df


def _engineer_features(df_in: pd.DataFrame, feature_fit_cutoff: pd.Timestamp) -> pd.DataFrame:
    """Create causal covariate features and historical physics-feature proxies."""
    df = df_in.copy().sort_values("time").reset_index(drop=True)
    t0 = df["time"].iloc[0]
    df["timestamp_days"] = (df["time"] - t0).dt.total_seconds() / 86400.0

    # Match the primary model: a trailing 7-observation moving average. The main script
    # separately warns if observation intervals are not approximately daily.
    df["temperature_7d"] = df["temperature"].rolling(window=7, min_periods=1).mean().ffill()
    df["water_level_7d"] = df["water_level"].rolling(window=7, min_periods=1).mean().ffill()

    day_of_year = df["time"].dt.dayofyear.to_numpy(dtype=float)
    day_of_month = df["time"].dt.day.to_numpy(dtype=float)
    df["year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * day_of_month / 30.4375)
    df["month_cos"] = np.cos(2 * np.pi * day_of_month / 30.4375)

    # Physics/engineering feature proxies are derived ONLY from the same environmental
    # inputs available to the other covariate-driven models. Do not derive input features
    # from displacement labels: that would give the feature baseline access to target data
    # unavailable to the unconstrained LSTM / Transformer and risk target leakage.
    dh = df["water_level"].diff()
    d2h = dh.diff()
    dt = df["temperature"].diff()
    eps = 1e-8
    df["phy_mono"] = np.sign(dh).fillna(0.0)             # water-level trend direction proxy
    df["phy_hydro"] = dh.fillna(0.0)                     # water-level change
    df["phy_thermal"] = df["temperature"] - df["temperature_7d"]
    df["phy_accel"] = d2h.fillna(0.0)                    # second difference of water level
    train_mask = df["time"] < feature_fit_cutoff
    if not train_mask.any():
        raise ValueError("No training rows exist before TRAIN_END_DATE")
    water_q80 = float(df.loc[train_mask, "water_level"].quantile(0.80))
    df["phy_flood"] = (df["water_level"] >= water_q80).astype(float)
    df["phy_rate"] = dh.abs().fillna(0.0)                # magnitude of loading change
    df["phy_hooke"] = (dh.abs() / (dt.abs() + eps)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # A water-level-change proxy only; this is NOT measured seepage pressure.
    df["phy_seepage_proxy"] = (df["water_level"] - df["water_level"].shift(2)).fillna(0.0)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].ffill().fillna(0.0)
    return df


@dataclass
class PreparedData:
    df: pd.DataFrame
    feature_scaler: RobustScaler
    physics_scaler: RobustScaler
    target_scaler: RobustScaler
    feature_matrix: np.ndarray
    physics_matrix: np.ndarray
    target_norm: np.ndarray
    train_target_idx: np.ndarray
    val_target_idx: np.ndarray
    test_target_idx: np.ndarray
    seq_len: int
    train_cutoff: pd.Timestamp
    val_cutoff: pd.Timestamp


def prepare_data(data_dir: str | Path,
                 seq_len: int = DEFAULT_SEQ_LEN,
                 train_end: str = DEFAULT_TRAIN_END,
                 start_date: str = DEFAULT_START,
                 end_date: str = DEFAULT_END,
                 val_fraction: float = 0.20) -> PreparedData:
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2")
    raw = load_raw_data(data_dir, start_date=start_date, end_date=end_date)
    cutoff = pd.Timestamp(train_end)
    start_ts = pd.Timestamp(start_date)
    if cutoff <= start_ts:
        raise ValueError("train_end must be later than start_date")
    # Use the same calendar-time validation boundary as the primary PhyLSTM script:
    # validation is the final val_fraction of the nominal pre-test training period.
    val_cutoff_for_features = start_ts + pd.to_timedelta((cutoff - start_ts).total_seconds() / 86400.0 * (1 - val_fraction), unit="D")
    n_pretest_rows = int((raw["time"] < cutoff).sum())
    if n_pretest_rows <= seq_len + 10:
        raise ValueError("Training period is too short for the requested sequence length")
    df = _engineer_features(raw, val_cutoff_for_features)
    times = df["time"]
    n_train_rows = int((times < cutoff).sum())
    scaler_fit_end = int((times < val_cutoff_for_features).sum())
    if scaler_fit_end <= seq_len + 5 or scaler_fit_end >= n_train_rows:
        raise ValueError("Chronological validation split leaves too few training/validation rows")

    # Fit scalers only on the gradient-training subset (before the validation boundary).
    feature_scaler = RobustScaler().fit(df.loc[:scaler_fit_end - 1, [
        "timestamp_days", "temperature", "water_level", "temperature_7d", "water_level_7d",
        "year_sin", "year_cos", "month_sin", "month_cos"
    ]])
    phys_cont_cols = ["phy_mono", "phy_hydro", "phy_thermal", "phy_accel", "phy_rate", "phy_hooke", "phy_seepage_proxy"]
    physics_scaler = RobustScaler().fit(df.loc[:scaler_fit_end - 1, phys_cont_cols])
    target_scaler = RobustScaler().fit(df.loc[:scaler_fit_end - 1, ["displacement"]])

    feature_source = df[[
        "timestamp_days", "temperature", "water_level", "temperature_7d", "water_level_7d",
        "year_sin", "year_cos", "month_sin", "month_cos"
    ]]
    feature_matrix = feature_scaler.transform(feature_source).astype(np.float32)

    phys_raw = np.zeros((len(df), 8), dtype=np.float64)
    phys_raw[:, 0:4] = df[["phy_mono", "phy_hydro", "phy_thermal", "phy_accel"]].to_numpy(dtype=np.float64)
    phys_raw[:, 4] = df["phy_flood"].to_numpy(dtype=np.float64)
    phys_raw[:, 5:8] = df[["phy_rate", "phy_hooke", "phy_seepage_proxy"]].to_numpy(dtype=np.float64)
    phys_matrix = np.zeros_like(phys_raw, dtype=np.float64)
    # Binary flood indicator is not standardized; the remaining continuous variables are.
    transformed_cont = physics_scaler.transform(df[phys_cont_cols])
    phys_matrix[:, :4] = transformed_cont[:, :4]
    phys_matrix[:, 5:8] = transformed_cont[:, 4:7]
    phys_matrix[:, 4] = phys_raw[:, 4]
    phys_matrix = phys_matrix.astype(np.float32)

    target_norm = target_scaler.transform(df[["displacement"]]).reshape(-1).astype(np.float32)

    # Targets are row t; their inputs comprise rows t-seq_len through t-1. No current target
    # row is ever included in its own input window. Test targets start at the held-out date,
    # while their input windows may use the preceding observed history as operational context.
    all_idx = np.arange(seq_len, len(df), dtype=np.int64)
    time_values = df["time"].to_numpy()
    train_target_idx = all_idx[time_values[all_idx] < val_cutoff_for_features.to_datetime64()]
    val_target_idx = all_idx[(time_values[all_idx] >= val_cutoff_for_features.to_datetime64()) &
                             (time_values[all_idx] < cutoff.to_datetime64())]
    test_target_idx = all_idx[time_values[all_idx] >= cutoff.to_datetime64()]
    if min(map(len, (train_target_idx, val_target_idx, test_target_idx))) < 1:
        raise ValueError(
            f"One split is empty (train={len(train_target_idx)}, val={len(val_target_idx)}, "
            f"test={len(test_target_idx)}). Check dates and sequence length."
        )
    return PreparedData(
        df=df, feature_scaler=feature_scaler, physics_scaler=physics_scaler,
        target_scaler=target_scaler, feature_matrix=feature_matrix,
        physics_matrix=phys_matrix, target_norm=target_norm,
        train_target_idx=train_target_idx, val_target_idx=val_target_idx,
        test_target_idx=test_target_idx, seq_len=seq_len, train_cutoff=cutoff,
        val_cutoff=val_cutoff_for_features,
    )


class WindowDataset(Dataset):
    def __init__(self, data: PreparedData, target_indices: np.ndarray, use_physics: bool = False):
        self.data = data
        self.indices = np.asarray(target_indices, dtype=np.int64)
        self.use_physics = use_physics

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        target_idx = int(self.indices[item])
        start = target_idx - self.data.seq_len
        base = self.data.feature_matrix[start:target_idx]
        if self.use_physics:
            x = np.concatenate([base, self.data.physics_matrix[start:target_idx]], axis=1)
        else:
            x = base
        y = self.data.target_norm[target_idx]
        return torch.from_numpy(x.copy()), torch.tensor([y], dtype=torch.float32)


def make_loader(data: PreparedData, indices: np.ndarray, *, use_physics=False, batch_size=32, shuffle=False):
    dataset = WindowDataset(data, indices, use_physics=use_physics)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


class LastStepLSTM(nn.Module):
    def __init__(self, input_size=9, hidden_size=128, num_layers=2, bidirectional=False, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0.0)
        out_size = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Linear(out_size, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        sequence, _ = self.lstm(x)
        return self.head(sequence[:, -1, :])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"Sequence length {x.size(1)} exceeds positional encoding maximum")
        return x + self.pe[:, :x.size(1)]


class VanillaTransformer(nn.Module):
    def __init__(self, input_size=9, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        h = self.encoder(self.pos(self.proj(x)))
        return self.head(h[:, -1, :])


class PhysicsFeatureLSTM(nn.Module):
    """LSTM fed with 9 base features plus 8 causal physical/engineering feature proxies."""
    def __init__(self, input_size=17, hidden_size=128, num_layers=2):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=0.3 if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        x = self.input_norm(x)
        h, _ = self.lstm(x)
        return self.head(h[:, -1, :])


class PhysicsEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 32), nn.LayerNorm(32), nn.GELU(),
                                 nn.Linear(32, d_model), nn.LayerNorm(d_model))
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


class PhysicsGuidedAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model, self.nhead, self.dk = d_model, nhead, d_model // nhead
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.wp_q = nn.Linear(d_model, d_model)
        self.wp_k = nn.Linear(d_model, d_model)
        self.alpha_raw = nn.Parameter(torch.tensor(-6.0))
        self.dropout = nn.Dropout(dropout)
        self.scale = self.dk ** 0.5

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_raw)

    def forward(self, x, physics):
        b, length, dim = x.shape
        q = self.wq(x).view(b, length, self.nhead, self.dk).transpose(1, 2)
        k = self.wk(x).view(b, length, self.nhead, self.dk).transpose(1, 2)
        v = self.wv(x).view(b, length, self.nhead, self.dk).transpose(1, 2)
        score = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        ph = F.normalize(physics, dim=-1)
        pq = self.wp_q(ph).view(b, length, self.nhead, self.dk).transpose(1, 2)
        pk = self.wp_k(ph).view(b, length, self.nhead, self.dk).transpose(1, 2)
        physics_score = (torch.matmul(pq, pk.transpose(-2, -1)) / self.scale).clamp(-2, 2) / 4.0
        attention = self.dropout(torch.softmax(score + self.alpha * physics_score, dim=-1))
        out = torch.matmul(attention, v).transpose(1, 2).reshape(b, length, dim)
        return self.wo(out)


class PhysicsTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.attn = PhysicsGuidedAttention(d_model, nhead, dropout)
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.residual_scale = 0.5

    def forward(self, x, physics):
        x = x + self.residual_scale * self.drop1(self.attn(self.norm1(x), physics))
        x = x + self.residual_scale * self.drop2(self.ffn(self.norm2(x)))
        return x


class PhysicsGuidedTransformer(nn.Module):
    def __init__(self, input_size=17, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        if input_size != 17:
            raise ValueError("PhysicsGuidedTransformer expects 9 base + 8 physics features (17 total)")
        self.proj = nn.Linear(input_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.physics_encoder = PhysicsEncoder(d_model)
        self.blocks = nn.ModuleList([
            PhysicsTransformerBlock(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        base = self.pos(self.proj(x))
        physics = self.physics_encoder(x[:, :, 9:])
        h = base
        for block in self.blocks:
            h = block(h, physics)
        return self.head(h[:, -1, :])


class PINNLSTM(nn.Module):
    def __init__(self, input_size=9, hidden_size=128, num_layers=2):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=0.3 if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, 1)
        # Positive softplus-constrained coefficients for the empirical physical residual.
        self.raw_alpha = nn.Parameter(torch.tensor(0.0))
        self.raw_beta = nn.Parameter(torch.tensor(0.0))
        self.raw_gamma = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x):
        h, _ = self.lstm(self.input_norm(x))
        return self.head(h[:, -1, :])

    def physics_loss(self, x, pred, h0, t0, d0):
        # Gradient w.r.t. the latest observed time coordinate. Inputs must require grad.
        grad = torch.autograd.grad(pred.sum(), x, create_graph=True, retain_graph=True)[0]
        dy_dt = grad[:, -1, 0:1]
        temp = x[:, -1, 1:2]
        water = x[:, -1, 2:3]
        rhs = F.softplus(self.raw_alpha) * (water - h0) + F.softplus(self.raw_beta) * (temp - t0) - F.softplus(self.raw_gamma) * (pred - d0)
        residual = dy_dt - rhs
        return F.smooth_l1_loss(residual, torch.zeros_like(residual))


class HSTRegressor:
    """A transparent HST-style least-squares baseline with hydrostatic, seasonal, and time terms.

    This replaces the supplied script's generic linear regression, which was labelled HST
    but lacked an explicit hydrostatic polynomial and time component. Values will therefore
    differ from that previous implementation and must be recomputed if used in the paper.
    """
    def __init__(self):
        self.scaler = RobustScaler()
        self.regressor = LinearRegression()

    def fit(self, x, y):
        xs = self.scaler.fit_transform(x)
        self.regressor.fit(xs, y)

    def predict(self, x):
        return self.regressor.predict(self.scaler.transform(x))


class ARIMAForecaster:
    def __init__(self, order=(5, 1, 0)):
        self.order = order
        self.fit_result = None

    def fit(self, y):
        if ARIMA is None:
            raise ImportError("ARIMA requires statsmodels. Install the requirements in requirements.txt")
        self.fit_result = ARIMA(np.asarray(y, dtype=float), order=self.order).fit()

    def forecast(self, steps: int) -> np.ndarray:
        return np.asarray(self.fit_result.forecast(steps=steps), dtype=float).reshape(-1)


def make_model(model_name: str, input_size: int):
    key = model_name.lower().replace("-", "_").replace(" ", "_")
    if key in {"lstm", "vanilla_lstm", "unconstrained_lstm"}:
        return LastStepLSTM(input_size=input_size)
    if key in {"bilstm", "bi_lstm"}:
        return LastStepLSTM(input_size=input_size, bidirectional=True)
    if key in {"transformer", "vanilla_transformer"}:
        return VanillaTransformer(input_size=input_size)
    if key in {"pinn_lstm", "pinn"}:
        return PINNLSTM(input_size=input_size)
    if key in {"physics_feature_lstm", "physics_lstm"}:
        return PhysicsFeatureLSTM(input_size=input_size)
    if key in {"physics_guided_transformer", "pg_transformer"}:
        return PhysicsGuidedTransformer(input_size=input_size)
    raise ValueError(f"Unknown neural model: {model_name}")


def _torch_predict(model, loader, device, target_scaler, model_name, h0=None, t0=None, d0=None):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            preds.append(out.detach().cpu().numpy().reshape(-1))
            trues.append(y.numpy().reshape(-1))
    pred_norm = np.concatenate(preds)
    true_norm = np.concatenate(trues)
    return target_scaler.inverse_transform(pred_norm.reshape(-1, 1)).reshape(-1), \
           target_scaler.inverse_transform(true_norm.reshape(-1, 1)).reshape(-1)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_true) != len(y_pred):
        raise ValueError(f"Prediction/target length mismatch: {len(y_pred)} vs {len(y_true)}")
    if len(y_true) < 2:
        raise ValueError("At least two test observations are required for evaluation")
    drange = float(np.max(y_true) - np.min(y_true))
    tau = kendalltau(y_true, y_pred).statistic
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "nMAPE": float(np.mean(np.abs(y_pred - y_true)) / drange * 100) if drange > 1e-12 else float("nan"),
        "Kendall_tau": float(tau) if tau is not None and np.isfinite(tau) else float("nan"),
    }


def _dataset_for_model(data: PreparedData, target_indices: np.ndarray, model_name: str, batch_size: int, shuffle: bool):
    use_physics = model_name.lower().replace("-", "_").replace(" ", "_") in {
        "physics_feature_lstm", "physics_lstm", "physics_guided_transformer", "pg_transformer"
    }
    loader = make_loader(data, target_indices, use_physics=use_physics, batch_size=batch_size, shuffle=shuffle)
    input_size = 17 if use_physics else 9
    return loader, input_size, use_physics


def train_neural_model(data: PreparedData, model_name: str, output_dir: str | Path,
                       epochs: int = 100, batch_size: int = 32, learning_rate: float = 5e-4,
                       seed: int = 42, physics_weight: float = 0.1, patience: int = 25):
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, input_size, use_physics = _dataset_for_model(data, data.train_target_idx, model_name, batch_size, True)
    val_loader, _, _ = _dataset_for_model(data, data.val_target_idx, model_name, batch_size, False)
    test_loader, _, _ = _dataset_for_model(data, data.test_target_idx, model_name, batch_size, False)

    model = make_model(model_name, input_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                           patience=max(3, patience // 3), min_lr=1e-6)

    # RobustScaler centers are training-only quantities. Use the normalized training means
    # for the empirical PINN-LSTM residual reference values.
    n_grad_train_rows = int((data.df["time"] < data.val_cutoff).sum())
    h0 = torch.tensor(float(np.mean(data.feature_matrix[:n_grad_train_rows, 2])), device=device)
    t0 = torch.tensor(float(np.mean(data.feature_matrix[:n_grad_train_rows, 1])), device=device)
    d0 = torch.tensor(float(np.mean(data.target_norm[:n_grad_train_rows])), device=device)

    best_val = float("inf")
    best_state = None
    history: List[dict] = []
    wait = 0
    start_clock = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if isinstance(model, PINNLSTM):
                x.requires_grad_(True)
                pred = model(x)
                data_loss = F.mse_loss(pred, y)
                ramp = min(1.0, max(0.0, (epoch + 1) / max(1, min(30, epochs))))
                phy_loss = model.physics_loss(x, pred, h0, t0, d0)
                loss = data_loss + physics_weight * ramp * phy_loss
            else:
                pred = model(x)
                loss = F.mse_loss(pred, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss encountered in {model_name}, epoch {epoch+1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_losses.append(float(F.mse_loss(model(x), y).cpu()))
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{model_name}: epoch {epoch+1}/{epochs} train={train_loss:.6f} val={val_loss:.6f}")
        if wait >= patience:
            print(f"Early stopping {model_name} at epoch {epoch+1}; best validation loss={best_val:.6f}")
            break

    train_seconds = time.perf_counter() - start_clock
    if best_state is None:
        raise RuntimeError(f"Training did not produce a valid state for {model_name}")
    model.load_state_dict(best_state)
    # One final test evaluation only, after model selection is completed on validation data.
    pred, truth = _torch_predict(model, test_loader, device, data.target_scaler, model_name)
    test_times = data.df["time"].iloc[data.test_target_idx].to_numpy()
    metric = compute_metrics(truth, pred)
    metric.update({
        "model": model_name,
        "train_seconds": train_seconds,
        "n_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "best_val_mse_normalized": best_val,
        "best_epoch": int(min(history, key=lambda row: row["val_loss"])["epoch"]),
        "test_samples": int(len(truth)),
    })
    np.savez_compressed(output_dir / f"{model_slug(model_name)}_predictions.npz",
                        time=test_times.astype(str), y_true=truth, y_pred=pred)
    pd.DataFrame({"time": test_times, "y_true": truth, "y_pred": pred}).to_csv(
        output_dir / f"{model_slug(model_name)}_predictions.csv", index=False
    )
    pd.DataFrame(history).to_csv(output_dir / f"{model_slug(model_name)}_training_history.csv", index=False)
    torch.save(best_state, output_dir / f"{model_slug(model_name)}_best.pt")
    return metric, np.asarray(test_times), truth, pred


def _target_indices_by_time(data: PreparedData, target_indices: np.ndarray):
    return np.asarray(target_indices, dtype=int)


def run_hst(data: PreparedData, output_dir: str | Path):
    """HST-style hydrostatic-seasonal-time regression at a one-step forecast origin.

    Predict target y_t from physical covariates available at t-1 and the calendar terms
    for t. Uses train-only fitting and validation for protocol checks; final fit uses all
    pre-test targets. The equation differs from the generic regression originally labelled
    HST, so resulting metrics must be recomputed rather than reused from prior tables.
    """
    frame = data.df
    tr_idx = data.train_target_idx
    va_idx = data.val_target_idx
    te_idx = data.test_target_idx
    model = HSTRegressor()
    hst_start = time.perf_counter()

    def x_for_targets(indices):
        # water level from the last observed input row; calendar terms for target date are deterministic.
        previous = indices - 1
        times = frame["time"].iloc[indices].reset_index(drop=True)
        water = frame["water_level"].to_numpy(dtype=float)[previous]
        time0 = pd.Timestamp(frame["time"].iloc[0])
        years = np.array([(pd.Timestamp(t) - time0).total_seconds() / (365.25 * 86400) for t in times])
        doy = np.array([pd.Timestamp(t).dayofyear for t in times], dtype=float)
        phase = 2 * np.pi * doy / 365.25
        return np.column_stack([water, water**2, water**3, np.sin(phase), np.cos(phase),
                                np.sin(2*phase), np.cos(2*phase), years, np.log1p(np.maximum(years, 0))])

    y_raw = frame["displacement"].to_numpy(dtype=float)
    model.fit(x_for_targets(tr_idx), y_raw[tr_idx])
    val_pred = model.predict(x_for_targets(va_idx))
    val_mse = mean_squared_error(y_raw[va_idx], val_pred)
    # No tuning is performed on the validation set. Refit on every observation before test.
    pretest_idx = np.concatenate([tr_idx, va_idx])
    model.fit(x_for_targets(pretest_idx), y_raw[pretest_idx])
    pred = model.predict(x_for_targets(te_idx))
    truth, times = y_raw[te_idx], frame["time"].iloc[te_idx].to_numpy()
    hst_seconds = time.perf_counter() - hst_start
    metrics = compute_metrics(truth, pred)
    metrics.update({"model": "HST", "train_seconds": hst_seconds, "n_parameters": int(len(model.regressor.coef_) + 1),
                    "best_val_mse_normalized": float("nan"), "validation_mse_raw": float(val_mse),
                    "best_epoch": 0, "test_samples": len(truth)})
    _write_prediction_artifacts(output_dir, "HST", times, truth, pred)
    return metrics, times, truth, pred


def run_arima(data: PreparedData, output_dir: str | Path, order=(5, 1, 0)):
    if ARIMA is None:
        raise ImportError("ARIMA requires statsmodels; install requirements.txt first")
    frame = data.df
    te_idx = data.test_target_idx
    # Open-loop forecast, fitted only to data observed before the test interval.
    pretest_y = frame["displacement"].iloc[:te_idx[0]].to_numpy(dtype=float)
    test_y = frame["displacement"].to_numpy(dtype=float)[te_idx]
    start = time.perf_counter()
    result = ARIMA(pretest_y, order=order).fit()
    pred = np.asarray(result.forecast(steps=len(te_idx)), dtype=float).reshape(-1)
    elapsed = time.perf_counter() - start
    times = frame["time"].iloc[te_idx].to_numpy()
    metrics = compute_metrics(test_y, pred)
    metrics.update({"model": "ARIMA", "train_seconds": elapsed, "n_parameters": int(len(result.params)),
                    "best_val_mse_normalized": float("nan"), "best_epoch": 0, "test_samples": len(test_y)})
    _write_prediction_artifacts(output_dir, "ARIMA", times, test_y, pred)
    return metrics, times, test_y, pred


def model_slug(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def _write_prediction_artifacts(output_dir, name, times, truth, pred):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time": times, "y_true": truth, "y_pred": pred}).to_csv(out / f"{model_slug(name)}_predictions.csv", index=False)


def run_models(data_dir: str | Path, output_dir: str | Path, models: Sequence[str], seq_len: int = DEFAULT_SEQ_LEN,
               train_end: str = DEFAULT_TRAIN_END, start_date: str = DEFAULT_START,
               end_date: str = DEFAULT_END, epochs: int = 100, batch_size: int = 32,
               learning_rate: float = 5e-4, seed: int = 42, physics_weight: float = 0.1,
               val_fraction: float = 0.20):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading and preparing data (scalers are fitted on gradient-training interval only)...")
    data = prepare_data(data_dir, seq_len=seq_len, train_end=train_end, start_date=start_date,
                        end_date=end_date, val_fraction=val_fraction)
    print(f"Rows={len(data.df)} | train targets={len(data.train_target_idx)} | "
          f"validation targets={len(data.val_target_idx)} | test targets={len(data.test_target_idx)}")
    print(f"Training cutoff={data.train_cutoff.date()} | validation cutoff={data.val_cutoff.date()} | sequence={seq_len}")

    all_metrics = []
    for name in models:
        name_norm = name.lower().replace("-", "_").replace(" ", "_")
        if name_norm == "hst":
            result = run_hst(data, output_dir)
        elif name_norm == "arima":
            result = run_arima(data, output_dir)
        else:
            result = train_neural_model(data, name, output_dir, epochs=epochs, batch_size=batch_size,
                                        learning_rate=learning_rate, seed=seed,
                                        physics_weight=physics_weight)
        metrics, _, _, _ = result
        all_metrics.append(metrics)
        print(f"Finished {metrics['model']}: R2={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")

    metrics_df = pd.DataFrame(all_metrics)
    if len(metrics_df):
        cols_first = ["model", "R2", "MAE", "RMSE", "nMAPE", "Kendall_tau", "train_seconds", "n_parameters", "best_val_mse_normalized", "best_epoch", "test_samples"]
        metrics_df = metrics_df[[c for c in cols_first if c in metrics_df.columns] + [c for c in metrics_df.columns if c not in cols_first]]
    metrics_df.to_csv(output_dir / "comparison_metrics.csv", index=False)
    metrics_df.to_excel(output_dir / "comparison_metrics.xlsx", index=False)

    # Join the exported prediction series on timestamps for a single Origin-friendly table.
    merged = None
    for name in models:
        p = output_dir / f"{model_slug(name)}_predictions.csv"
        if not p.exists():
            continue
        frame = pd.read_csv(p, parse_dates=["time"])
        if merged is None:
            merged = frame.rename(columns={"y_true": "actual_displacement", "y_pred": name})[["time", "actual_displacement", name]]
        else:
            merged = merged.merge(frame[["time", "y_pred"]].rename(columns={"y_pred": name}), on="time", how="outer", validate="one_to_one")
    if merged is not None:
        merged.sort_values("time").to_csv(output_dir / "all_model_predictions_for_origin.csv", index=False)
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "seq_len": seq_len, "train_end_exclusive": train_end, "start_date": start_date,
            "end_date_inclusive": end_date, "validation_fraction_of_pretest_period": val_fraction,
            "seed": seed, "epochs": epochs, "batch_size": batch_size,
            "models": list(models),
            "feature_information_notes": {
                "base_models": "Use environmental, calendar, and trailing environmental features only.",
                "physics_feature_lstm_and_physics_guided_transformer": "Use 8 environmental/engineering feature transformations derived from temperature, water level, and calendar covariates; no displacement labels are input features.",
                "arima": "Univariate open-loop displacement forecast; input information differs from covariate-driven neural models.",
                "hst": "HST-style hydrostatic-seasonal-time least-squares baseline; replaces previous generic regression labelled HST."
            },
            "data_policy": "No raw dam monitoring data are bundled."
        }, f, ensure_ascii=False, indent=2)
    return metrics_df


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train/evaluate dam-deformation comparison models")
    parser.add_argument("--data-dir", required=True, help="Directory containing original dam data folders")
    parser.add_argument("--output-dir", default="comparison_outputs", help="Directory for metrics and predictions")
    parser.add_argument("--models", nargs="+", default=["lstm", "pinn_lstm", "physics_feature_lstm", "physics_guided_transformer", "hst", "arima", "bilstm", "transformer"],
                        help="Model names: lstm pinn_lstm physics_feature_lstm physics_guided_transformer hst arima bilstm transformer")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--train-end", default=DEFAULT_TRAIN_END, help="Exclusive test-period start date")
    parser.add_argument("--start-date", default=DEFAULT_START)
    parser.add_argument("--end-date", default=DEFAULT_END)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--physics-weight", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    args = parser.parse_args(argv)
    run_models(args.data_dir, args.output_dir, args.models, seq_len=args.seq_len,
               train_end=args.train_end, start_date=args.start_date, end_date=args.end_date,
               epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
               seed=args.seed, physics_weight=args.physics_weight, val_fraction=args.val_fraction)


if __name__ == "__main__":
    main()

