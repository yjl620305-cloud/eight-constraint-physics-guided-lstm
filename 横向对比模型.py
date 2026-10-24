# -*- coding: utf-8 -*-
"""Horizontal benchmark: HST, ARIMA, BiLSTM, and vanilla Transformer."""
from run_single_model import run_wrapper

if __name__ == "__main__":
    run_wrapper(["HST", "ARIMA", "bilstm", "transformer"])

