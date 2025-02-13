"""
barycenter_mnist.py
-------------------

Compare the barycenter of different digits using our method and the true barycenter.
"""

import torch
import ot
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.evaluation.import_models as im
import src.utils.data_functions as df
import src.ot.cost_matrix as cost
from src.evaluation.barycenter import barycenter


def plot_barycenter(predictor, dimension, device):
    with torch.no_grad():
        mnist = torch.load("data/mnist.pt")

    mnist = df.preprocessor(mnist, dimension, 1e-6)

    cost_matrix = cost.fast_get_cost_matrix(64, device)

    liste6 = [0, 7, 9, 19, 27, 32]
    liste1 = [6, 15, 16, 45, 44, 31, 84]
    liste8 = [2, 4, 11, 23, 28, 36, 60]
    liste0 = [1, 30, 33, 40, 47, 52, 59, 71, 72, 75, 87, 136, 150, 151, 158, 164, 183]
    listen_mist = [liste6, liste1, liste8, liste0]

    bar = []
    true_bar = []
    for liste in listen_mist:
        A = barycenter(predictor, mnist[liste].to(device), cost_matrix, nits=200)
        bar.append(A)
        B = ot.bregman.barycenter_debiased(
            mnist[liste].T, cost_matrix.cpu(), 0.01, numItermax=2000
        )
        true_bar.append(B)

    fig, axs = plt.subplots(2, 4, figsize=(8, 4))

    # Add the left-side labels
    labels = ["Ours", "True"]
    for i, label in enumerate(labels):
        axs[i, 0].text(
            -0.2,
            0.5,
            label,
            va="center",
            ha="right",
            rotation=90,
            fontsize=22,
            transform=axs[i, 0].transAxes,
        )

    # Time labels for the columns
    # Plot True Geodesic
    for i, x in enumerate(bar):
        axs[0, i].axis("off")  # Hide axes
        axs[0, i].imshow(
            x.reshape(dimension, dimension),
            cmap="gray",
        )

    # Plot Plan Geodesic
    for i, x in enumerate(true_bar):
        axs[1, i].axis("off")  # Hide axes
        axs[1, i].imshow(
            x.reshape(dimension, dimension),
            cmap="gray",
        )

    # Adjust the layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("Images/mnist_barycenter.pdf")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Barycenter shapes")
    parser.add_argument("--model", type=str, default="UNOT", help="Model to use")
    parser.add_argument(
        "--dimension", type=int, default=64, help="Dimension of the images"
    )
    return parser.parse_args()


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    args = parse_args()
    predictor = im.load_fno(args.model, device=device)
    plot_barycenter(predictor, 64, device)


if __name__ == "__main__":
    # Load the models
    main()
