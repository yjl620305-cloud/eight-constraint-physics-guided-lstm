# -*- coding: utf-8 -*-
"""Run all comparison models under the same time split and forecast-target alignment."""
import sys
from comparison_models import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--models" not in args:
        args = args + ["--models", "lstm", "pinn_lstm", "physics_feature_lstm",
                       "physics_guided_transformer", "hst", "arima", "bilstm", "transformer"]
    main(args)

