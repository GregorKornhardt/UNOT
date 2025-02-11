import torch
import os
import sys
import argparse
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.train.train_var_eps as train
import src.ot.cost_matrix as cost
from src.networks.FNO2d import FNO2d
from src.networks.generator  import Generator

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for optimal transport")

    # Experiment & General Settings
    parser.add_argument('--name', type=str, default='NO_var_eps', help='Experiment name')
    parser.add_argument('--dtype', type=str, default='float32', help='Data type')
    parser.add_argument('--wandb', type=bool, default=False, help='Use wandb for logging')

    # Data & Input Settings
    parser.add_argument('--length', type=int, default=64, help='Length of input data')
    parser.add_argument('--length_latent', type=int, default=10, help='Length of latent space')
    parser.add_argument('--numbr_latent_samples', type=int, default=100, help='Number of latent samples')

    # Generator Architecture
    parser.add_argument('--generator', type=str, default='MLP', help='Generator model type')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in generator')
    parser.add_argument('--skip_connection', type=float, default=1, help='Skip connection value in Generator')
    parser.add_argument('--width_generator', type=int, default=4, help='Width of generator network, as a multiple of data length')

    # Neural Operator (FNO) Architecture
    parser.add_argument('--model', type=str, default='FNO', help='Neural operator model type')
    parser.add_argument('--modes_x', type=int, default=20, help='Fourier modes in x')
    parser.add_argument('--modes_y', type=int, default=20, help='Fourier modes in y')
    parser.add_argument('--fno_width', type=int, default=64, help='Width of FNO')
    parser.add_argument('--fno_blocks', type=int, default=4, help='Number of FNO Blocks')
    parser.add_argument('--n_hidden_spectral_layers', type=int, default=2, help='Number of hidden layers in spectral network')
    parser.add_argument('--sobel', type=bool, default=False, help='Use Sobel filter')
    parser.add_argument('--grid', type=bool, default=False, help='Use grid data')

    # Regularization & Weight Decay
    parser.add_argument('--weight_decay_generator', type=float, default=0., help='Weight decay for generator')
    parser.add_argument('--weight_decay_predictor', type=float, default=1e-4, help='Weight decay for predictor')
    parser.add_argument('--dust_const', type=float, default=1e-6, help='Dust constant')
    parser.add_argument('--dust', type=float, default=1e-4, help='Dust value')

    # Optimization & Training
    parser.add_argument('--lr_gen', type=float, default=0.01, help='Learning rate for generator')
    parser.add_argument('--lr_pred', type=float, default=1e-4, help='Learning rate for predictor')
    parser.add_argument('--gamma_generator', type=float, default=0.99996, help='Gamma for generator LR scheduler')
    parser.add_argument('--gamma_predictor', type=float, default=0.99996, help='Gamma for predictor LR scheduler')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--nits', type=int, default=400, help='Number of training iterations')
    parser.add_argument('--nits_mini_loop_generator', type=int, default=1, help='Number of mini loops for generator')
    parser.add_argument('--nits_mini_loop_predictor', type=int, default=1, help='Number of mini loops for predictor')

    # Sinkhorn Algorithm Parameters
    parser.add_argument('--sinkhorn_max_iterations', type=int, default=50, help='Max iterations for Sinkhorn algorithm')
    parser.add_argument('--sinkhorn_epsilon', type=float, default=1e-2, help='Epsilon for Sinkhorn algorithm')

    # Data Settings
    parser.add_argument('--use_data', type=bool, default=False, help='Use additional data during training')

    return parser.parse_args()

def neural_operator_input_channel(args):
    if args.sobel:    
        input_chanel = 9
    elif args.grid:
        input_chanel = 5
    else:
        input_chanel = 3
    return input_chanel


def set_up_wandb(args):
    if args.wandb is False:
        wandb.init(mode="disabled")
    wandb.init(
        # set the wandb project where this run will be logged
        project="fno-ot",

        # track hyperparameters and run metadata
        config=vars(args)
    )
    run_name = wandb.run.name
    return run_name


def get_dtype(args):
    if args.dtype == 'float64':
        torch.set_default_dtype(torch.float64)
        dtype = torch.float64
    else: 
        torch.set_default_dtype(torch.float32)
        dtype = torch.float32
    return dtype


def main():
    args = parse_args()
    dir_prefix = '.'
    dtype = get_dtype(args)
    
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    print('Using device:', device)
    
    run_name = set_up_wandb(args)

    # PROBLEM HYPERPARAMETERS
    dimesion_latent = args.length_latent ** 2
    dimension = args.length ** 2
    dim_generator = int(args.width_generator * args.length**2)

    # initialize networks
    input_chanel = neural_operator_input_channel(args)
        
    if args.generator == 'MLP':
        generator64 = Generator(dim_prior=dimesion_latent, dim=dimension, dim_hidden=dim_generator, 
                                       num_layers=args.num_layers, dust_const=args.dust_const,
                                       skip_const=1).to(device)
    else:
        raise NotImplementedError
    if args.model == 'FNO':
        predictor = FNO2d(input_chanel,1,(args.modes_x, args.modes_y), args.fno_width, 
                              activation=torch.nn.GELU(), n_blocks=args.fno_blocks).to(device)

    predictor.train()
    generator64.train()

    # load test data
    data_set = []
    if args.use_data:
        raise NotImplementedError

    try:        
        test_data_28 = torch.load(f"{dir_prefix}/Data/test_set__dim_28__eps_0.01.pt", weights_only=True)
        test_data_64 = torch.load(f"{dir_prefix}/Data/test_set__dim_64__eps_0.01.pt", weights_only=True)
    except:
        test_data_28 = None
        test_data_64 = None

    # load cost matrix
    cost_matrix_28 = cost.get_cost_matrix(28, device).to(dtype)
    cost_matrix_64 = cost.get_cost_matrix(64, device).to(dtype)
    
    # create Models directory if it doesn't exist
    os.makedirs(f"{dir_prefix}/Models", exist_ok=True)

    # Train the model
    trainer = train.Training(
                            generator64,
                            predictor,
                            lr_gen = args.lr_gen,
                            lr_fno = args.lr_pred,
                            weight_decay_generator = args.weight_decay_generator,
                            weight_decay_predictor = args.weight_decay_predictor,
                            multiplicative_factor_generator = args.gamma_generator,
                            multiplicative_factor_predictor = args.gamma_predictor,
                            numbr_training_iterations = args.nits,
                            numbr_mini_loop_predictor = args.nits_mini_loop_predictor,
                            numbr_mini_loop_generator = args.nits_mini_loop_generator,
                            sinkhorn_max_iterations = args.sinkhorn_max_iterations,
                            sinkhorn_epsilon = args.sinkhorn_epsilon,
                            cost_matrix_28 = cost_matrix_28,
                            cost_matrix_64=cost_matrix_64,
                            numbr_latent_samples=args.numbr_latent_samples,
                            dust_const=args.dust_const,
                            length = args.length,
                            batch_size = args.batch_size,
                            dim_prior = dimesion_latent,
                            use_data=args.use_data,
                            which_gen = args.generator,
                            grid=args.grid,
                            sobel=args.sobel,
                            test_data_28 = test_data_28,
                            test_data_64 = test_data_64,
                            path = dir_prefix,
                            name = run_name,
                            data_set=data_set,
                            device = device)

    trainer.train() 
    
    # save model
    torch.save(trainer.predictor.state_dict(), f"{dir_prefix}/Models/fno_{args.length}_{run_name}_{run_name}.pt")
    if test_data_64 is not None and test_data_28 is not None:
        torch.save(trainer.best_model, f"{dir_prefix}/Models/fno_best_model_{args.length}_{run_name}_{run_name}.pt")


if __name__ == "__main__":
    main()