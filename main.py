"""
Main entry point for Project 2: Denoising Autoencoders for 1D Time-Series.

Usage:
    python main.py                     # run all experiments
    python main.py --exp arch          # architecture comparison only
    python main.py --exp latent        # latent dimension sweep
    python main.py --exp noise         # noise robustness
    python main.py --exp hyperparam    # hyperparameter search
"""

import argparse
import os

import config
from dataset import generate_synthetic_signals, build_dataloaders
from noise import make_noise_fn
from experiments import (
    run_architecture_comparison,
    run_latent_dim_experiment,
    run_noise_robustness_experiment,
    run_hyperparameter_search,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        default="all",
        choices=["all", "arch", "latent", "noise", "hyperparam"],
        help="Which experiment to run",
    )
    parser.add_argument("--epochs",  type=int,   default=config.EPOCHS)
    parser.add_argument("--lr",      type=float, default=config.LEARNING_RATE)
    parser.add_argument("--batch",   type=int,   default=config.BATCH_SIZE)
    parser.add_argument("--results", type=str,   default=config.RESULTS_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results, exist_ok=True)

    print("Generating synthetic signals ...")
    signals = generate_synthetic_signals(
        n_samples=5000, signal_length=config.SIGNAL_LENGTH
    )

    print(f"Building dataloaders (batch={args.batch}) ...")
    train_loader, val_loader, test_loader = build_dataloaders(
        signals, batch_size=args.batch
    )

    noise_fn = make_noise_fn("gaussian", sigma=config.GAUSSIAN_SIGMA)

    run_all = args.exp == "all"

    if run_all or args.exp == "hyperparam":
        print("\n" + "=" * 60)
        print("EXPERIMENT: Hyperparameter Search")
        print("=" * 60)
        run_hyperparameter_search(
            signals, noise_fn,
            arch="cnn",
            results_dir=args.results,
            epochs=min(args.epochs, 50),
        )

    if run_all or args.exp == "arch":
        print("\n" + "=" * 60)
        print("EXPERIMENT: Architecture Comparison (FC / CNN / LSTM)")
        print("=" * 60)
        run_architecture_comparison(
            train_loader, val_loader, test_loader, noise_fn,
            results_dir=args.results,
            epochs=args.epochs,
            lr=args.lr,
        )

    if run_all or args.exp == "latent":
        print("\n" + "=" * 60)
        print("EXPERIMENT: Latent Dimension Sweep")
        print("=" * 60)
        run_latent_dim_experiment(
            train_loader, val_loader, test_loader, noise_fn,
            results_dir=args.results,
            epochs=args.epochs,
            lr=args.lr,
        )

    if run_all or args.exp == "noise":
        print("\n" + "=" * 60)
        print("EXPERIMENT: Noise Robustness")
        print("=" * 60)
        run_noise_robustness_experiment(
            train_loader, val_loader, test_loader,
            train_sigma=config.GAUSSIAN_SIGMA,
            results_dir=args.results,
            epochs=args.epochs,
            lr=args.lr,
        )

    print("\nAll experiments complete. Results saved to:", args.results)


if __name__ == "__main__":
    main()
