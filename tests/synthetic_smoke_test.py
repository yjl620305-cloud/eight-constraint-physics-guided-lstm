# -*- coding: utf-8 -*-
"""Fast synthetic data-flow check. This does NOT test performance on real dam data."""
from pathlib import Path
import sys
import tempfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from comparison_models import (
    PreparedData, train_neural_model, run_hst, run_arima,
)


def build_synthetic_prepared_data(n=150, seq_len=14, train_end_idx=110, val_end_idx=90):
    rng = np.random.default_rng(0)
    times = pd.date_range("2012-08-13", periods=n, freq="D")
    idx = np.arange(n, dtype=float)
    temp = 12 + 10 * np.sin(2 * np.pi * idx / 365.25) + rng.normal(0, 0.2, n)
    water = 1640 + 2 * np.sin(2 * np.pi * idx / 30) + rng.normal(0, 0.1, n)
    displacement = 0.02 * idx + 0.12 * np.sin(2 * np.pi * idx / 21) + 0.002 * temp + rng.normal(0, 0.01, n)
    frame = pd.DataFrame({"time": times, "temperature": temp, "water_level": water, "displacement": displacement})
    frame["timestamp_days"] = idx
    frame["temperature_7d"] = frame["temperature"].rolling(7, min_periods=1).mean()
    frame["water_level_7d"] = frame["water_level"].rolling(7, min_periods=1).mean()
    frame["year_sin"] = np.sin(2*np.pi*times.dayofyear/365.25)
    frame["year_cos"] = np.cos(2*np.pi*times.dayofyear/365.25)
    frame["month_sin"] = np.sin(2*np.pi*times.day/30.4375)
    frame["month_cos"] = np.cos(2*np.pi*times.day/30.4375)
    d1 = frame["displacement"].diff()
    d2 = d1.diff()
    dh = frame["water_level"].diff()
    dt = frame["temperature"].diff()
    frame["phy_mono"] = d1.fillna(0)
    frame["phy_hydro"] = dh.fillna(0)
    frame["phy_thermal"] = frame["temperature"] - frame["temperature_7d"]
    frame["phy_accel"] = d2.fillna(0)
    frame["phy_flood"] = (frame["water_level"] >= frame.loc[:val_end_idx-1, "water_level"].quantile(.8)).astype(float)
    frame["phy_rate"] = d1.abs().fillna(0)
    frame["phy_hooke"] = (d1.abs() / (dt.abs() + dh.abs() + 1e-8)).replace([np.inf, -np.inf], np.nan).fillna(0)
    frame["phy_seepage_proxy"] = (frame["water_level"] - frame["water_level"].shift(2)).fillna(0)

    feature_cols = ["timestamp_days", "temperature", "water_level", "temperature_7d", "water_level_7d",
                    "year_sin", "year_cos", "month_sin", "month_cos"]
    physical_cols = ["phy_mono", "phy_hydro", "phy_thermal", "phy_accel", "phy_rate", "phy_hooke", "phy_seepage_proxy"]
    scaler_x = RobustScaler().fit(frame.loc[:val_end_idx-1, feature_cols])
    scaler_p = RobustScaler().fit(frame.loc[:val_end_idx-1, physical_cols])
    scaler_y = RobustScaler().fit(frame.loc[:val_end_idx-1, ["displacement"]])
    feature_matrix = scaler_x.transform(frame[feature_cols]).astype(np.float32)
    p_trans = scaler_p.transform(frame[physical_cols])
    phys_matrix = np.zeros((n, 8), dtype=np.float32)
    phys_matrix[:, :4] = p_trans[:, :4]
    phys_matrix[:, 4] = frame["phy_flood"].to_numpy(dtype=np.float32)
    phys_matrix[:, 5:] = p_trans[:, 4:]
    target_norm = scaler_y.transform(frame[["displacement"]]).ravel().astype(np.float32)
    all_idx = np.arange(seq_len, n)
    return PreparedData(
        df=frame, feature_scaler=scaler_x, physics_scaler=scaler_p, target_scaler=scaler_y,
        feature_matrix=feature_matrix, physics_matrix=phys_matrix, target_norm=target_norm,
        train_target_idx=all_idx[all_idx < val_end_idx],
        val_target_idx=all_idx[(all_idx >= val_end_idx) & (all_idx < train_end_idx)],
        test_target_idx=all_idx[all_idx >= train_end_idx], seq_len=seq_len,
        train_cutoff=times[train_end_idx], val_cutoff=times[val_end_idx],
    )


def main():
    data = build_synthetic_prepared_data()
    with tempfile.TemporaryDirectory(prefix="phylstm_smoke_") as tempdir:
        out = Path(tempdir)
        for model in ["lstm", "pinn_lstm", "physics_feature_lstm", "physics_guided_transformer", "bilstm", "transformer"]:
            metrics, *_ = train_neural_model(data, model, out, epochs=1, batch_size=16, patience=1, seed=7)
            assert np.isfinite(metrics["RMSE"]), f"Non-finite metrics from {model}"
            print(f"PASS {model}")
        for function in [run_hst, run_arima]:
            metrics, *_ = function(data, out)
            assert np.isfinite(metrics["RMSE"]), f"Non-finite metrics from {metrics['model']}"
            print(f"PASS {metrics['model']}")
    print("Synthetic smoke test passed. These synthetic metrics are not manuscript results.")


if __name__ == "__main__":
    main()

