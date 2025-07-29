#!/usr/bin/env python3
"""
Training script for SFNO optimal transport with argparse and wandb
"""

import os
import sys
import time
import argparse

# ensure project src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import wandb
import matplotlib.pyplot as plt

import src.networks.SFNO as SFNO
import src.networks.SFNO_small as SFNO_small
import src.networks.generator as networks
import src.train.train_sfno as train
import src.ot.cost_matrix as cost
import src.utils.data_functions as df

# default directory prefix for datasets, models, and outputs
DIR_PREFIX = "/n/netscratch/dam_lab/Lab/jgeuter/OTinit/Optimal-Transport"


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for SFNO-based optimal transport on the sphere.")
    # Job configuration
    parser.add_argument("--task-id", type=int, required=False, default=0, help="Array job task ID (unused when using direct parser args)")
    parser.add_argument("--array-job-id", type=int, required=True, help="SLURM array job ID or similar")
    parser.add_argument("--project", type=str, required=True, help="WandB project name")
    parser.add_argument("--test", action="store_true", help="Run in test mode (disable wandb, use local dirs)")

    # Experiment & Naming
    parser.add_argument("--name", type=str, default="", help="Experiment name prefix")

    # Data & Input Settings
    parser.add_argument("--length", type=int, default=64, help="Length of input data")
    parser.add_argument("--length-latent", type=int, default=10, help="Length of latent space")

    # Generator Architecture
    parser.add_argument("--generator", type=str, default="MLP", help="Generator model type (MLP or FNO)")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of layers in generator")
    parser.add_argument("--width-generator", type=float, default=0.04, help="Width of generator network as fraction of data")
    parser.add_argument("--dust-const", type=float, default=1e-6, help="Dust constant for generator")
    parser.add_argument("--dust", type=float, default=1e-4, help="Dust value for generator")

    # Neural Operator (SFNO) Architecture
    parser.add_argument("--model", type=str, default="FNO_small", help="SFNO model type (FNO or FNO_small)")
    parser.add_argument("--modes-x", type=int, default=10, help="Fourier modes in x")
    parser.add_argument("--modes-y", type=int, default=10, help="Fourier modes in y")
    parser.add_argument("--fno-width", type=int, default=64, help="Width of SFNO layers")
    parser.add_argument("--fno-blocks", type=int, default=2, help="Number of SFNO blocks")
    parser.add_argument("--spectral-hidden-width", type=int, default=4, help="Hidden width for spectral network")
    parser.add_argument("--n-hidden-spectral-layers", type=int, default=2, help="Number of hidden spectral layers")
    parser.add_argument("--sobel", action="store_true", help="Use Sobel filter on inputs")
    parser.add_argument("--grid", action="store_true", help="Use grid data channels")

    # Regularization & Weight Decay
    parser.add_argument("--weight-decay-generator", type=float, default=0.0, help="Weight decay for generator")
    parser.add_argument("--weight-decay-predictor", type=float, default=1e-4, help="Weight decay for predictor")

    # Optimization & Training
    parser.add_argument("--lr-gen", type=float, default=0.001, help="Learning rate for generator")
    parser.add_argument("--lr-pred", type=float, default=1e-4, help="Learning rate for predictor")
    parser.add_argument("--gamma-generator", type=float, default=1.0, help="LR gamma for generator")
    parser.add_argument("--gamma-predictor", type=float, default=0.99996, help="LR gamma for predictor")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--nits", type=int, default=40000, help="Total training iterations")
    parser.add_argument("--nits-mini-loop-generator", type=int, default=1, help="Mini-loop iterations for generator per step")
    parser.add_argument("--nits-mini-loop-predictor", type=int, default=1, help="Mini-loop iterations for predictor per step")

    # Sinkhorn Algorithm Parameters
    parser.add_argument("--sinkhorn-max-iterations", type=int, default=5, help="Max iterations for Sinkhorn algorithm")
    parser.add_argument("--sinkhorn-epsilon", type=float, default=1e-2, help="Epsilon for Sinkhorn algorithm")

    # Latent samples & data flags
    parser.add_argument("--numbr-latent-samples", type=int, default=5000, help="Number of latent samples for generator")
    parser.add_argument("--use-data", action="store_true", help="Use additional dataset during training")
    parser.add_argument("--adam-neuralop", action="store_true", help="Use Adam optimizer for neural operator")

    return parser.parse_args()


def neural_operator_input_channel(args) -> int:
    """Determine number of input channels based on preprocessing flags."""
    if args.sobel:
        return 8
    elif args.grid:
        return 4
    else:
        return 2


def get_dtype(dtype_str: str):
    """Set torch default dtype and return torch dtype."""
    if dtype_str == 'float64':
        torch.set_default_dtype(torch.float64)
        return torch.float64
    else:
        torch.set_default_dtype(torch.float32)
        return torch.float32


def set_up_wandb(args, exp_name: str, dir_prefix: str) -> str:
    """Initialize wandb and return run name."""
    if args.test:
        wandb.init(mode="disabled")
    else:
        os.makedirs(f"{dir_prefix}/wandb", exist_ok=True)
        wandb.init(
            project=args.project,
            name=exp_name,
            id=exp_name,
            resume="allow",
            config=vars(args),
            dir=f"{dir_prefix}/wandb"
        )
    return wandb.run.name


def main():
    args = parse_args()

    # derive name with date suffix
    name_date = f"{args.name}_{time.strftime('%Y-%m-%d')}" if args.name else time.strftime('%Y-%m-%d')

    # determine root directory
    dir_prefix = '.' if args.test else DIR_PREFIX

    # set dtype and device
    dtype = get_dtype(args.dtype)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    # initialize wandb
    exp_name = f"{args.array_job_id}_{args.array_job_id}_sfno"
    run_name = set_up_wandb(args, exp_name, dir_prefix)

    # compute dimensions
    dlatent = args.length_latent ** 2
    dimension = args.length ** 2
    dgen = int(args.width_generator * dimension)

    # build generator
    if args.generator == 'MLP':
        generator = networks.Generator(
            dim_prior=dlatent,
            dim=dimension,
            dim_hidden=dgen,
            num_layers=args.num_layers,
            dust_const=args.dust_const,
            skip_const=1
        ).to(device)
    else:
        generator = SFNO.FNO2d(
            2, 2,
            (args.modes_x, args.modes_y),
            args.fno_width,
            activation=torch.nn.GELU(),
            n_blocks=args.fno_blocks
        ).to(device)
    generator.train()

    # build predictor
    in_ch = neural_operator_input_channel(args)
    if args.model == 'FNO':
        predictor = SFNO.SFNO3d(
            in_ch, 1,
            (args.modes_x, args.modes_y),
            args.fno_width,
            activation=torch.nn.GELU(),
            n_blocks=args.fno_blocks,
            spectral_hidden_width=args.spectral_hidden_width
        ).to(device)
    else:
        predictor = SFNO_small.SFNO3d(
            in_ch, 1,
            (args.modes_x, args.modes_y),
            args.fno_width,
            activation=torch.nn.GELU(),
            n_blocks=args.fno_blocks,
            spectral_hidden_width=args.spectral_hidden_width
        ).to(device)
        total_p = sum(p.numel() for p in predictor.parameters())
        train_p = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
        print(f'FNO total parameters: {total_p}, trainable parameters: {train_p}')
    predictor.train()

    # load test data
    try:
        test_data_28 = torch.load(f"{dir_prefix}/Data/test_set__dim_28__eps_0.01.pt", weights_only=True)
        test_data_64 = torch.load(f"{dir_prefix}/Data/test_set__dim_64__eps_0.01.pt", weights_only=True)
    except FileNotFoundError:
        test_data_28 = test_data_64 = None

    # load cost matrices
    cost28 = cost.get_cost_matrix_S2(28, device).to(dtype)
    cost64 = cost.get_cost_matrix_S2(64, device).to(dtype)

    # prepare output dirs
    os.makedirs(f"{dir_prefix}/Images", exist_ok=True)
    os.makedirs(f"{dir_prefix}/Models", exist_ok=True)

    # initialize trainer and run training
    trainer = train.Training(
        generator,
        predictor,
        lr_gen=args.lr_gen,
        lr_fno=args.lr_pred,
        weight_decay_generator=args.weight_decay_generator,
        weight_decay_predictor=args.weight_decay_predictor,
        multiplicative_factor_generator=args.gamma_generator,
        multiplicative_factor_predictor=args.gamma_predictor,
        numbr_training_iterations=args.nits,
        numbr_mini_loop_predictor=args.nits_mini_loop_predictor,
        numbr_mini_loop_generator=args.nits_mini_loop_generator,
        sinkhorn_max_iterations=args.sinkhorn_max_iterations,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        cost_matrix_28=cost28,
        cost_matrix_64=cost64,
        numbr_latent_samples=args.numbr_latent_samples,
        dust_const=args.dust_const,
        length=args.length,
        batch_size=args.batch_size,
        dim_prior=dlatent,
        use_data=args.use_data,
        which_gen=args.generator,
        grid=args.grid,
        sobel=args.sobel,
        test_data_28=test_data_28,
        test_data_64=test_data_64,
        data_set=[],
        name=exp_name,
        path=dir_prefix,
        device=device
    )
    trainer.train()

    # save models
    torch.save(
        trainer.predictor.state_dict(),
        f"{dir_prefix}/Models/fno_{args.length}_{name_date}_{run_name}.pt"
    )
    if trainer.best_model is not None:
        torch.save(
            trainer.best_model,
            f"{dir_prefix}/Models/fno_best_model_{args.length}_{name_date}_{run_name}.pt"
        )
    if hasattr(trainer, 'model_save') and trainer.model_save is not None:
        torch.save(
            trainer.model_save,
            f"{dir_prefix}/Models/fno_model_save_{args.length}_{name_date}_{run_name}.pt"
        )


if __name__ == "__main__":
    main()
