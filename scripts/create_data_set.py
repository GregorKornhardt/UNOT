import argparse
import os 
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.utils.data_functions as df

def main():
    parser = argparse.ArgumentParser(description='Create data set for optimal transport.')
    parser.add_argument('--num_elements', type=int, default=300, help='Number of elements')
    parser.add_argument('--length', type=int, default=28, help='Length of the data')
    parser.add_argument('--dust_const', type=float, default=1e-6, help='Dust constant')
    parser.add_argument('--epsilon', type=float, default=1e-2, help='Epsilon value')
    parser.add_argument('--numb_per_data_set', type=int, default=10000, help='Number per data set')
    parser.add_argument('--sinkhorn_iter', type=int, default=100, help='Number of Sinkhorn iterations')
    parser.add_argument('--device', type=str, default='mps', help='Device to use')
    parser.add_argument('--incl_random_shapes', type=bool, default=True, help='Include random shapes')

    args = parser.parse_args()

    df.create_data_set_grf(
        args.num_elements,
        args.numb_per_data_set,
        args.length,
        args.dust_const,
        args.epsilon,
        args.sinkhorn_iter,
        args.incl_random_shapes,
        args.device
    )

if __name__ == '__main__':
    main()