"""
barycenter_shapes.py
-------

This script will generate a barycenter of different shapes and save the image.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
import argparse
import os 
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.evaluation.import_models as im
import src.ot.cost_matrix  as cost
from src.evaluation.barycenter import barycenter


def shape_measures(
        dimension, 
        device
    ):
    def donut():
        x, y = np.meshgrid(
            np.linspace(-2, 2, dimension), np.linspace(-2, 2, dimension)
        )

        # Define the donut shape function
        r = np.sqrt(x**2 + y**2)  # Radial distance
        donut = (r - 1)**2 - 0.2  # Donut equation

        donut[(donut >= 0)] = 0
        donut[donut != 0] = 1
        donut = torch.tensor(donut.flatten()).float().to(device) - donut.min()
        donut /= donut.sum()
        donut += 1e-6
        donut = donut / donut.sum()
        return donut

    def two_circles():
        matrix_circles = np.zeros((dimension, dimension))

        # Create the circles
        radius = 10
        center1 = (15, 15)  # Circle 1 center
        center2 = (45, 45)  # Circle 2 center

        for i in range(dimension):
            for j in range(dimension):
                # Check if the pixel lies within either circle
                if (i - center1[0])**2 + (j - center1[1])**2 < radius**2:
                    matrix_circles[i, j] = 1
                if (i - center2[0])**2 + (j - center2[1])**2 < radius**2:
                    matrix_circles[i, j] = 1
        
        return matrix_circles

    def X():
        matrix_x = np.zeros((dimension, dimension))

        # Create the "X" shape
        thickness = 10  # Thickness of the lines
        for i in range(dimension):
            for j in range(dimension):
                if abs(i - j) < thickness or abs(i + j - dimension) < thickness:
                    matrix_x[i, j] = 1
        return matrix_x
    
    def heart():
        matrix_heart = np.zeros((dimension, dimension))

        # Heart parametric equation mapped to the grid
        x_center, y_center = dimension // 2, dimension // 2
        scale = 20  # Scaling factor for the heart size

        for i in range(dimension):
            for j in range(dimension):
                # Map matrix indices to the heart equation
                x = (i - x_center) / scale
                y = (j - y_center) / scale
                # Heart equation: (x² + y² - 1)³ - x²y³ <= 0
                if (x**2 + y**2 - 1)**3 - x**2 * y**3 <= 0:
                    matrix_heart[i, j] = 1

        matrix_heart = np.rot90(matrix_heart, k=1)

        return matrix_heart
    
    matrix_heart_tensor = torch.tensor(heart().copy()).float().to(device)
    matrix_circles_tensor = torch.tensor(two_circles().copy()).float().to(device)
    matrix_x_tensor = torch.tensor(X().copy()).float().to(device)

    mu = torch.stack([matrix_heart_tensor.flatten(), donut(), matrix_circles_tensor.flatten(), matrix_x_tensor.flatten()]).to(device)
    mu /= mu.sum(1).reshape(-1,1)
    mu += 1e-6
    mu /= mu.sum(1).reshape(-1,1)

    return mu


def plot_barycenter(
        predictor, 
        device
    ):
    mu = shape_measures(64, device)
    cost_matrix = cost.fast_get_cost_matrix(64, device)

    # Dimensions for the small images (n, n)
    n = 64  # Example size of each small image (64x64)

    # Number of images per row and column in the large image
    images_per_row = 5
    images_per_column = 5

    # Create an empty array to hold the large image
    large_image = np.zeros((n * (images_per_column) , n * images_per_row))

    # Reshape each image from (batch, n*n) to (n, n) and place it in the large image
    x_offset = 0
    y_offset = 0
    
    def get_weights():
        M = np.zeros((5, 5, 4), dtype=float)

        for i in range(5):
            for j in range(5):
                x = i / 4.0
                y = j / 4.0
                M[i, j] = [(1 - x) * (1 - y),
                        x * (1 - y),
                        (1 - x) * y,
                        x * y]
        return M
    
    M = get_weights()

    for k in range(5):
            for i in range(5):
                if y_offset >= n * images_per_column:
                    break
                with torch.no_grad():
                    bar = barycenter(predictor, mu, cost_matrix, weights=M[k, i]).cpu()
                small_image = bar.reshape(n, n)
                
                large_image[y_offset:y_offset + n, x_offset:x_offset + n] = small_image

                # Update the position for the next image
                x_offset += n
                if x_offset >= n * images_per_row:
                    x_offset = 0
                    y_offset += n

    large_image = (large_image - large_image.min()) / (large_image.max() - large_image.min()) * 255

    def create_color_matrix(n):
        # Define more saturated base colors
        R = np.array([1, 0, 0])
        G = np.array([0, 1, 0])
        Y = np.array([1, 0.8, 0])  # Slightly more saturated yellow
        B = np.array([0, 0, 1])
        
        M = np.zeros((n, n, 3), dtype=float)
        for i in range(n):
            for j in range(n):
                x = i / (n - 1)
                y = j / (n - 1)
                # Use power function to make transitions more abrupt
                x = np.power(x, 1.5)
                y = np.power(y, 1.5)
                M[i, j] = (1-x)*(1-y)*R + x*(1-y)*G + (1-x)*y*Y + x*y*B
                
                # Increase saturation
                max_val = np.max(M[i, j])
                if max_val > 0:
                    M[i, j] = np.power(M[i, j]/max_val, 0.7) * max_val
        
        return M
    
    colormatrix = create_color_matrix(5)
    
    def create_custom_colormap(color):
        # Define colors for our custom colormap
        colors = [color, 'white']
        
        # Create custom colormap
        n_bins = 256  # Number of discrete color levels
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        
        return cmap


    # Number of images per row and column in the large image
    images_per_row = 5
    images_per_column = 5

    # Create an empty array to hold the large image
    fig, axs = plt.subplots(5, 5, figsize=(5,5))

    # Reshape each image from (batch, n*n) to (n, n) and place it in the large image
    x_offset = 0
    y_offset = 0
    for k in range(5):
            for i in range(5):
                if y_offset >= n * images_per_column:
                    break

                small_image = large_image[y_offset:y_offset + n, x_offset:x_offset + n]
                
                cmap = create_custom_colormap(colormatrix[k,i])
                axs[k,i].imshow(1-small_image, cmap=cmap)
                axs[k, i].axis('off')
                # Update the position for the next image
                x_offset += n
                if x_offset >= n * images_per_row:
                    x_offset = 0
                    y_offset += n

    plt.tight_layout(pad=0.1)
    plt.savefig('Images/color_image_barycenter.pdf', bbox_inches='tight', pad_inches=0.1)


def parse_args():
    parser = argparse.ArgumentParser(description='Barycenter shapes')
    parser.add_argument('--model', type=str, default='unot', help='Model to use')
    parser.add_argument('--dimension', type=int, default=64, help='Dimension of the images')
    return parser.parse_args()


def main():
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")

    args = parse_args()
    predictor = im.load_fno(args.model, device=device)
    plot_barycenter(predictor, device)


if __name__=='__main__':
    # Load the models
    main()