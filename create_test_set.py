import argparse

import src.utils.data_functions as df

def main():
    parser = argparse.ArgumentParser(description='Create a test set for optimal transport.')
    parser.add_argument('--num_elements', type=int, default=600, help='Number of elements in the test set')
    parser.add_argument('--length', type=int, default=64, help='Length of each element')
    parser.add_argument('--dust_const', type=float, default=1e-6, help='Dust constant')
    parser.add_argument('--epsilon', type=float, default=5e-2, help='Epsilon value')
    parser.add_argument('--device', type=str, default='mps', help='Device to use')

    args = parser.parse_args()

    df.create_test_set(args.num_elements, args.length, args.dust_const, args.epsilon, args.device)

if __name__ == '__main__':
    main()
