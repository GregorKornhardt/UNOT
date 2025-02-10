import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import Image
import math

import src.ot.cost_matrix  as cost
import src.ot.sinkhorn as sh
import src.utils.data_functions as df
import src.evaluation.import_models as im


class SD_outer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, predictor, mu, nu, x, y, cost_matrix):
        maxiter = 100
        # Compute f(x) using g(x) for the forward pass
        def inner_matrix(mu, nu):
            eps = 1e-2
            # OT(nu, nu)
            v0 = torch.exp(predictor(nu, nu))
            _,_,dist_y_ = sh.sink_vec_dist(nu, nu, cost_matrix, eps, v0, 1)
            dist_y = dist_y_.reshape(nu.size(0), -1)
            
            # OT(mu, mu)
            v0 = torch.exp(predictor(mu, mu))
            U_x,_,dist_x = sh.sink_vec_dist(mu, mu, cost_matrix, eps, v0, 1)
            dist_x = dist_x.reshape(-1, mu.size(0))
            U_x = U_x.unsqueeze(1).repeat(1, mu.size(0), 1)

            # OT(mu, nu)
            MU = mu.unsqueeze(1).repeat(1, nu.size(0), 1).reshape(-1, mu.size(1))
            NU = nu.unsqueeze(0).repeat(mu.size(0), 1, 1).reshape(-1, nu.size(1)) 
            v0 = torch.exp(predictor(MU, NU))

            _,V,dist = sh.sink_vec_dist(MU, NU, cost_matrix, eps, v0, 1)
            V = V.reshape(mu.size(0), nu.size(0), mu.size(1))
            dist = dist.reshape(mu.size(0), nu.size(0))
            
            # Compute the Sinkhorn divergence
            SD = dist - 0.5 * dist_x - 0.5 * dist_y
            V = 0.01 * torch.log(V)
            U_x = 0.01 * torch.log(U_x)

            return SD, V, U_x

        inner, V, U_x = inner_matrix(x, y)
        inner_x, V_x, U_x_x = inner_matrix(x, x)

        v0 = torch.ones_like(mu, device=mu.device)
        _,_,G,dist = sh.sink(mu, nu, inner, 1e-3, v0, maxiter)
        _,_,G_x,dist_x = sh.sink(mu, mu, inner_x, 1e-3, v0, maxiter)

        ctx.save_for_backward(G, G_x, V, U_x, V_x, U_x_x)
        return dist - 0.5 * dist_x
    
    @staticmethod
    def backward(ctx, grad_output):
        G, G_x, V, U_x, V_x, U_x_x = ctx.saved_tensors  
        
        g = torch.einsum("ij,ijn->in", G, V - U_x) - torch.einsum("ij,ijn->in", G_x, V_x - U_x_x) 
        
        grad_predictor = None   
        grad_mu = None
        grad_x =  - grad_output * g
        grad_y = None
        grad_nu = None
        grad_cost_matrix = None 
        return grad_predictor, grad_mu, grad_nu, grad_x, grad_y, grad_cost_matrix


def gradient_flow_image(predictor, y, niter=100):
    x = torch.rand_like(y, requires_grad=True) 
    images = []

    num_input = y.shape[0]
    mu = torch.ones(num_input, device=y.device) / num_input 
    nu = torch.ones(num_input, device=y.device) / num_input
    
    cost_matrix = cost.fast_get_cost_matrix(math.isqrt(y.shape[1]), device=y.device).to(y.device)
    softmax = torch.nn.Softmax()
    
    opt = torch.optim.Adam([x], lr=0.1)
    for i in tqdm(range(niter)):
        images.append(softmax(x).detach().cpu().numpy())
        dist = SD_outer.apply(predictor, mu, nu, softmax(x), y, cost_matrix)
        opt.zero_grad()
        dist.backward()
        opt.step()
        
    return x, images


def gradient_flow(predictor, y, niter=100):
    x = torch.rand_like(y, requires_grad=True) 

    num_input = y.shape[0]
    mu = torch.ones(num_input, device=y.device) / num_input 
    nu = torch.ones(num_input, device=y.device) / num_input
    
    cost_matrix = cost.fast_get_cost_matrix(math.isqrt(y.shape[1]), device=y.device).to(y.device)
    softmax = torch.nn.Softmax()
    
    opt = torch.optim.AdamW([x], lr=1e-1)
    for i in tqdm(range(niter)):
        dist = SD_outer.apply(predictor, mu, nu, softmax(x), y, cost_matrix)
        opt.zero_grad()
        dist.backward()
        opt.step()
        
    return x


def plot_all(images, ground_truth=None):
    # Dimensions for the small images (n, n)
    n = 64  # Example size of each small image (64x64)

    # Number of images per row and column in the large image
    images_per_row = images[0].shape[0]
    images_per_column = 10

    # Create an empty array to hold the large image
    large_image = np.zeros((n * (images_per_column+1) + 20 , n * images_per_row))

    # Reshape each image from (batch, n*n) to (n, n) and place it in the large image
    x_offset = 0
    y_offset = 0

    for k,batch in enumerate(images):
        if k % 10 == 0:
            for i in range(batch.shape[0]):
                if y_offset >= n * images_per_column:
                    break
                # Reshape each image into (n, n)
                small_image = (batch[i].reshape(n, n) - batch[i].min()) / (batch[i].max() - batch[i].min()) * 255
                # Place the reshaped image into the large image
                large_image[y_offset:y_offset + n, x_offset:x_offset + n] = small_image

                # Update the position for the next image
                x_offset += n
                if x_offset >= n * images_per_row:
                    x_offset = 0
                    y_offset += n
    
    for i in range(ground_truth.shape[0]):
        # Reshape each image into (n, n)
        index = (np.abs(images[-1][i]- ground_truth)).sum(1).argmin()
        small_image = (ground_truth[index].reshape(n, n) - ground_truth[index].min()) / (ground_truth[index].max() - ground_truth[index].min()) * 255
        # Place the reshaped image into the large image
        large_image[-1-n:-1, x_offset:x_offset + n] = small_image

        # Update the position for the next image
        x_offset += n       
    

    # Convert the large image into a PIL image for saving or displaying
    large_image_pil = Image.fromarray(np.uint8(large_image))

    # Save the large image
    large_image_pil.save('large_image.png')

    # Optional: Show the image
    large_image_pil.show()


def main():
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    
    # problem hyperparameters
    length = 64
    dust_const = 1e-6

    with torch.no_grad():
            lfw = torch.load('data/lfw.pt', weights_only=True)
            
    lfw = df.preprocessor(lfw, length, dust_const)
    
    predictor = im.load_model('predictor_64_eps=1e-2_2024-10-26', length**2, device)
    
    x,image = gradient_flow_image(predictor, lfw[2:12].to(device), 100)
    
    plt.show()
    plot_all(image, lfw[2:12].cpu().numpy())


if __name__ == '__main__':
    main()