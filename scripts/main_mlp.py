import torch
import os
import sys
import argparse
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.train.train_mlp as train
import src.ot.cost_matrix as cost
from src.networks.generator import Generator
from src.networks.mlp import Predictor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training script for optimal transport"
    )

    # Experiment & General Settings
    parser.add_argument("--name", type=str, default="eps=1e-2", help="Experiment name")
    parser.add_argument("--dtype", type=str, default="float32", help="Data type")
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=True,
        help="Enable or disable wandb (default: enabled)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_false",
        dest="wandb",
        help="Disable wandb explicitly",
    )

    # Data & Model Dimensions
    parser.add_argument("--length", type=int, default=28, help="Length of input data")
    parser.add_argument(
        "--length-latent", type=int, default=10, help="Length of latent space"
    )

    # Network Architecture
    parser.add_argument(
        "--num-layers", type=int, default=3, help="Number of layers in networks"
    )
    parser.add_argument(
        "--skip_connection",
        type=float,
        default=1,
        help="Skip connection value in Generator",
    )
    parser.add_argument(
        "--width-generator",
        type=float,
        default=4,
        help="Width of generator network, as multiple of data length",
    )
    parser.add_argument(
        "--width-predictor",
        type=float,
        default=4,
        help="Width of predictor network, as multiple of data length",
    )

    # Regularization & Weight Decay
    parser.add_argument(
        "--weight-decay-generator",
        type=float,
        default=0,
        help="Weight decay for generator",
    )
    parser.add_argument(
        "--weight-decay-predictor",
        type=float,
        default=1e-4,
        help="Weight decay for predictor",
    )
    parser.add_argument("--dust-const", type=float, default=1e-6, help="Dust constant")
    parser.add_argument("--dust", type=float, default=1e-4, help="Dust value")

    # Optimization & Training
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--gamma-generator",
        type=float,
        default=1.0,
        help="Gamma for generator learning rate scheduler",
    )
    parser.add_argument(
        "--gamma-predictor",
        type=float,
        default=0.9999,
        help="Gamma for predictor learning rate scheduler",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--nits", type=int, default=10000, help="Number of training iterations"
    )
    parser.add_argument(
        "--nits-mini-loop-generator",
        type=int,
        default=1,
        help="Number of mini loops for generator",
    )
    parser.add_argument(
        "--nits-mini-loop-predictor",
        type=int,
        default=1,
        help="Number of mini loops for predictor",
    )

    # Sinkhorn Algorithm Parameters
    parser.add_argument(
        "--sinkhorn-max-iterations",
        type=int,
        default=5,
        help="Max iterations for Sinkhorn algorithm",
    )
    parser.add_argument(
        "--sinkhorn-epsilon",
        type=float,
        default=1e-2,
        help="Epsilon for Sinkhorn algorithm",
    )

    return parser.parse_args()


def set_up_wandb(args):
    if args.wandb is False:
        wandb.init(mode="disabled")

    wandb.init(
        # set the wandb project where this run will be logged
        project="fno-ot",
        # track hyperparameters and run metadata
        config=vars(args),
    )
    run_name = wandb.run.name
    return run_name


def get_dtype(args):
    if args.dtype == "float64":
        raise NotImplementedError
    else:
        torch.set_default_dtype(torch.float32)
        dtype = torch.float32
    return dtype


def main():
    wandb.init(mode="disabled")
    args = parse_args()
    dir_prefix = "."

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    run_name = set_up_wandb(args)

    # HYPERPARAMETERS Generator
    dimesion_latent = args.length_latent**2
    dimension = args.length**2
    dim_generator = int(args.width_generator * args.length**2)
    dim_predictor = int(args.width_predictor * args.length**2)

    # initialize networks
    generator = Generator(
        dim_prior=dimesion_latent,
        dim=dimension,
        dim_hidden=dim_generator,
        num_layers=args.num_layers,
        dust_const=args.dust_const,
        skip_const=1,
    ).to(device)
    predictor = Predictor(dimension, dim_predictor, num_layers=args.num_layers).to(
        device
    )

    generator.train()
    predictor.train()

    # load test data
    try:
        if args.length == 28:
            test_data = torch.load(f"{dir_prefix}/Data/test_set__dim_28__eps_0.01.pt")
        elif args.length == 64:
            test_data = torch.load(f"{dir_prefix}/Data/test_set__dim_64__eps_0.01.pt")
    except:
        test_data = None

    # load cost matrix
    cost_matrix = cost.get_cost_matrix(args.length, device)

    # create Models directory if it doesn't exist
    os.makedirs(f"{dir_prefix}/Models", exist_ok=True)

    # Train the model
    trainer = train.Training(
        generator,
        predictor,
        lr=args.lr,
        weight_decay_generator=args.weight_decay_generator,
        weight_decay_predictor=args.weight_decay_predictor,
        multiplicative_factor_generator=args.gamma_generator,
        multiplicative_factor_predictor=args.gamma_predictor,
        numbr_training_iterations=args.nits,
        numbr_mini_loop_predictor=args.nits_mini_loop_predictor,
        numbr_mini_loop_generator=args.nits_mini_loop_generator,
        sinkhorn_max_iterations=args.sinkhorn_max_iterations,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        cost_matrix=cost_matrix,
        batch_size=args.batch_size,
        dim_prior=dimesion_latent,
        test_data=test_data,
        device=device,
    )

    trainer.train()

    # save model
    torch.save(
        trainer.predictor.state_dict(),
        f"{dir_prefix}/Models/predictor_{args.length}_{args.name}_{run_name}.pt",
    )


if __name__ == "__main__":
    main()
