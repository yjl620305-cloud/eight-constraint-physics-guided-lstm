# -*- coding: utf-8 -*-
"""Compatibility entry point used by the individual model scripts."""
import argparse
from comparison_models import run_models


def run_wrapper(model_names):
    parser = argparse.ArgumentParser(description=f"Run model(s): {', '.join(model_names)}")
    parser.add_argument("--data-dir", required=True, help="Directory containing original dam data folders")
    parser.add_argument("--output-dir", default="comparison_outputs")
    parser.add_argument("--seq-len", type=int, default=14)
    parser.add_argument("--train-end", default="2016-08-13")
    parser.add_argument("--start-date", default="2012-08-13")
    parser.add_argument("--end-date", default="2017-08-13")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--physics-weight", type=float, default=0.1)
    args = parser.parse_args()
    return run_models(
        data_dir=args.data_dir, output_dir=args.output_dir, models=model_names,
        seq_len=args.seq_len, train_end=args.train_end, start_date=args.start_date,
        end_date=args.end_date, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, seed=args.seed, physics_weight=args.physics_weight,
    )

