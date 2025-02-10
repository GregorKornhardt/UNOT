import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import wandb

def plot_complete_measures(mu, nu,run_name, save_path):
    """
    Plot measures for all classes
    :param measures: dictionary with measures for all classes
    :param title: title of the plot
    :param save_path: path to save the plot
    """
    # Create figure and grid spec
    fig = plt.figure(figsize=(18, 7.2))
    gs = gridspec.GridSpec(6,23, figure=fig, width_ratios=[1, 0.01, 1, 0.4, 1, 0.01, 1,0.4, 1, 0.01, 1, 0.4, 1, 0.01, 1, 0.4, 1, 0.01, 1, 0.4, 1, 0.01, 1], height_ratios=[1,1,1,1,1,1])
    gs.update(wspace=0.01, hspace=0.01)

    for i in range(6):
            if i == 1:
                ax.set_title(f'$\mu$ before\n training')
            ax = fig.add_subplot(gs[i, 0])
            ax.imshow(mu[0][i], cmap="gray")
            ax.axis('off')
    for i in range(6):
            if i == 1:
                ax.set_title(f'$\\nu$ before\n training')
            ax = fig.add_subplot(gs[i, 2])
            ax.imshow(nu[0][i], cmap="gray")
            ax.axis('off')

    # Plot first row
    for j in range(1, 6, 1):
        for i in range(6):
            if i == 1:
                ax.set_title(f'$\mu$ {j*10}%\n training')
            ax = fig.add_subplot(gs[i, 4*j])
            ax.imshow(mu[j][i], cmap="gray")
            ax.axis('off')
        for i in range(6):
            if i == 1:
                ax.set_title(f'$\\nu$ {j*10}%\n training')
            ax = fig.add_subplot(gs[i, 4*j+2])
            ax.imshow(nu[j][i], cmap="gray")
            ax.axis('off')
    wandb.log({"full_measure1": wandb.Image(fig, caption="Full Measure 1")})

    plt.savefig(save_path+'/'+run_name+'full_measure1.pdf', format='pdf')
    # Create figure and grid spec
    fig = plt.figure(figsize=(16, 7.2))
    gs = gridspec.GridSpec(6,19, figure=fig, width_ratios=[1, 0.01, 1, 0.4, 1, 0.01, 1,0.4, 1, 0.01, 1, 0.4, 1, 0.01, 1, 0.4, 1, 0.01, 1], height_ratios=[1,1,1,1,1,1])
    gs.update(wspace=0.01, hspace=0.01)

    
    # Plot first row
    for j in range(0, 4, 1):
        for i in range(6):
            if i == 1:
                ax.set_title(f'$\mu$ {j*10+60}%\n training')
            ax = fig.add_subplot(gs[i, 4*j])
            ax.imshow(mu[j][i], cmap="gray")
            ax.axis('off')
        for i in range(6):
            if i == 1:
                ax.set_title(f'$\\nu$ {j*10+60}%\n training')
            ax = fig.add_subplot(gs[i, 4*j+2])
            ax.imshow(nu[j][i], cmap="gray")
            ax.axis('off')

    for i in range(6):
            if i == 1:
                ax.set_title(f'$\mu$ after\n training')
            ax = fig.add_subplot(gs[i, 16])
            ax.imshow(mu[0][i], cmap="gray")
            ax.axis('off')
    for i in range(6):
            if i == 1:
                ax.set_title(f'$\\nu$ after\n training')
            ax = fig.add_subplot(gs[i, 18])
            ax.imshow(nu[0][i], cmap="gray")
            ax.axis('off')

    wandb.log({"full_measure2": wandb.Image(fig, caption="Full Measure 2")})
    plt.savefig(save_path+'/'+run_name+'full_measure2.pdf', format='pdf')

def plot_measure_short(mu, nu, run_name, save_path):
     # Create figure and grid spec
    fig = plt.figure(figsize=(8, 5))
    gs = gridspec.GridSpec(4,11, figure=fig, width_ratios=[1, 0.01, 1, 0.4, 1, 0.01, 1,0.4, 1, 0.01, 1], height_ratios=[1, 1,1, 1])
    gs.update(wspace=0.01, hspace=0.01)
    middle = len(mu)//2
    # Plot first row
    for i in range(4):
        if i == 1:
            ax.set_title(f'$\mu$ before\n training')
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(mu[0][i], cmap="gray")
        ax.axis('off')

    # Leave space for the middle row (row 1 is white space)

    # Plot third row
    for i in range(4):
        if i == 1:
            ax.set_title(f'$\\nu$ before\n training')
        ax = fig.add_subplot(gs[i, 2])
        ax.imshow(nu[0][i], cmap="gray")
        ax.axis('off')

    for i in range(4):
        if i == 1:
            ax.set_title(f'$\\nu$ during\n  training')
        ax = fig.add_subplot(gs[i, 4])
        ax.imshow(mu[middle][i], cmap="gray")
        ax.axis('off')
    for i in range(4):
        if i == 1:
            ax.set_title(f'$\\nu$ during\n training')
        ax = fig.add_subplot(gs[i, 6])
        ax.imshow(nu[middle][i], cmap="gray")
        ax.axis('off')

    for i in range(4):
        if i == 1:
            ax.set_title(f'$\mu$ during\n training')
        ax = fig.add_subplot(gs[i, 8])
        ax.imshow(mu[-1][i], cmap="gray")
        ax.axis('off')

    for i in range(4):
        if i == 1:
            ax.set_title(f'$\\nu$ after\n training')
        ax = fig.add_subplot(gs[i, 10])
        ax.imshow(nu[-1][i], cmap="gray")
        ax.axis('off')
    
    wandb.log({"short_measure": wandb.Image(fig, caption="Short Measure ")})
    plt.savefig(save_path+'/'+run_name+'short_measure.pdf', format='pdf')

def plot_measure_tight(mu, nu, run_name, save_path):
    # Create figure and grid spec
    rows, cols = 11, 11  # Number of rows and columns in the grid
    image_size = (64, 64)  # Size of each individual image

    # Create the figure
    fig, axes = plt.subplots(rows, cols, figsize=(11.5, 11.5), gridspec_kw={"wspace": 0.02, "hspace": 0.02})

    for i in range(rows):
        for j in range(cols):
            # Generate random noise image
            image = mu[i][j]
           
            # Plot the image
            axes[i, j].imshow(image, cmap="gray", aspect="auto")
            axes[i, j].axis("off")  # Remove axes for a clean look

    # Remove all padding
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    wandb.log({"measure_tight_mu": wandb.Image(fig, caption="Mu Measure")})

    plt.savefig(save_path+'/'+run_name+'full_mu.pdf', format='pdf')
    
    fig, axes = plt.subplots(rows, cols, figsize=(11.5, 11.5), gridspec_kw={"wspace": 0.02, "hspace": 0.02})

    for i in range(rows):
        for j in range(cols):
            # Generate random noise image
            image = nu[i][j]
            
            # Plot the image
            axes[i, j].imshow(image, cmap="gray", aspect="auto")
            axes[i, j].axis("off")  # Remove axes for a clean look

    # Remove all padding
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    wandb.log({"measure_tight_nu": wandb.Image(fig, caption="Mu Measure")})

    plt.savefig(save_path+'/'+run_name+'full_nu.pdf', format='pdf')
