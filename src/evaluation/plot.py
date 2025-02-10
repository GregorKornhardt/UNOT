import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D
import numpy as np

def plot_approximation(
        set_approx,
        model_name
    ):
    data_dict = {}
    for key in set_approx.keys():
        pred_set = torch.tensor(set_approx[key]['predicted'].cpu(), dtype= torch.float32)
        ones_set = torch.tensor(set_approx[key]['ones'].cpu(), dtype= torch.float32)
        gauss_set = torch.tensor(set_approx[key]['gauss'].cpu(), dtype= torch.float32)
        data_dict[key] = {'pred': pred_set, 'ones': ones_set, 'gauss': gauss_set}
    
    colors = {'pred': 'crimson', 'ones': 'indigo', 'gauss': 'cornflowerblue'}
    mean_colors = {'pred': 'crimson', 'ones': 'indigo', 'gauss': 'navy'}

    plt.figure(figsize=(5, 4))

    # Plot data
    for i, (label, datasets) in enumerate(data_dict.items(), start=1):
        for key, data in datasets.items():
            parts = plt.violinplot(data, positions=[i], showmeans=True, showmedians=False, showextrema=False)
            for pc in parts['bodies']:
                pc.set_facecolor(colors[key])
                pc.set_alpha(0.5)
            parts['cmeans'].set_color(mean_colors[key])

    # Set labels
    legend_entries = [
        plt.Line2D([0], [0], color=colors['ones'], lw=4, label='Ones'),
        plt.Line2D([0], [0], color=colors['gauss'], lw=4, label='Gauss'),
        plt.Line2D([0], [0], color=colors['pred'], lw=4, label='UNOT'),
    ]

    plt.legend(handles=legend_entries, loc='upper right')
    plt.grid(True)

    # X-Ticks
    print([key.upper() for key in data_dict.keys()])
    plt.xticks(range(1, len(data_dict) + 1), [key.upper() for key in data_dict.keys()])
    plt.ylabel('Relative Error')
    plt.yticks(torch.arange(0., .95, 0.05))

    plt.legend(handles=legend_entries, loc='upper right')
    plt.grid(True)

    # X-Ticks
    plt.xticks(range(1, len(data_dict) + 1),['MNIST', 'CIFAR', 'MNIST\nCIFAR', 'LFW-BEAR', 'LFW', 'BEAR'])

    # Gruppentitel hinzufügen
    group_labels = ['28x28', '28x28', '28x28', '64x64', '64x64', '64x64']
    for i, group in enumerate(group_labels, start=1):
        plt.text(i, -0.15, group, ha='center', va='top', fontsize=9, color='gray')  # Position unter den Labels

    plt.ylim(bottom=0)
    # Vermeide Überschneidungen
    plt.tight_layout()
    plt.savefig(f'Images/one_step_violin_plot_{model_name}.pdf')
    plt.show()


def plot_marginal_constraint(
        set_marginal,
        model_name
        ):
    
    data_dict = {}
    for key in set_marginal.keys():
        pred_set = torch.tensor(set_marginal[key]['predicted'], dtype= torch.float32)
        ones_set = torch.tensor(set_marginal[key]['ones'], dtype= torch.float32)
        gauss_set = torch.tensor(set_marginal[key]['gauss'], dtype= torch.float32)
        data_dict[key] = {'pred': pred_set, 'ones': ones_set, 'gauss': gauss_set}
    
    colors = {'pred': 'crimson', 'ones': 'indigo', 'gauss': 'cornflowerblue'}
    mean_colors = {'pred': 'crimson', 'ones': 'indigo', 'gauss': 'navy'}

    plt.figure(figsize=(5, 4))

    # Plot data
    for i, (label, datasets) in enumerate(data_dict.items(), start=1):
        for key, data in datasets.items():
            parts = plt.violinplot(data, positions=[i], showmeans=True, showmedians=False, showextrema=False)
            for pc in parts['bodies']:
                pc.set_facecolor(colors[key])
                pc.set_alpha(0.5)
            parts['cmeans'].set_color(mean_colors[key])

    # Set labels
    legend_entries = [
        plt.Line2D([0], [0], color=colors['ones'], lw=4, label='Ones'),
        plt.Line2D([0], [0], color=colors['gauss'], lw=4, label='Gauss'),
        plt.Line2D([0], [0], color=colors['pred'], lw=4, label='UNOT'),
    ]

    plt.legend(handles=legend_entries, loc='upper right')
    plt.grid(True)

    # X-Ticks
    plt.xticks(range(1, len(data_dict) + 1), ['MNIST', 'CIFAR', 'MNIST\nCIFAR', 'LFW-BEAR', 'LFW', 'BEAR'])

    # Gruppentitel hinzufügen
    group_labels = ['28x28', '28x28', '28x28', '64x64', '64x64', '64x64']
    for i, group in enumerate(group_labels, start=1):
        plt.text(i, -0.12, group, ha='center', va='top', fontsize=9, color='gray')  # Position unter den Labels

    plt.ylabel('Marginal Constraint Violation')
    plt.yticks(torch.arange(0, 0.65, 0.05))
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f'Images/violation_violin_plot_{model_name}.pdf')
    plt.show()




def print_iteration(
        set_iter_1,
        set_iter_2,
        set_iter_3,
):
    # Print the mean and std of the iterations for predicted and ones
    print(f"Predicted: {torch.mean(torch.tensor(set_iter_1['predicted'], dtype=torch.float32)).item()} +/- {torch.std(torch.tensor(set_iter_1['predicted'], dtype=torch.float32)).item()}")
    print(f"Ones: {torch.mean(torch.tensor(set_iter_1['ones'], dtype=torch.float32)).item()} +/- {torch.std(torch.tensor(set_iter_1['ones'], dtype=torch.float32)).item()}")
    print(f"Predicted: {torch.mean(torch.tensor(set_iter_2['predicted'], dtype=torch.float32)).item()} +/- {torch.std(torch.tensor(set_iter_2['predicted'], dtype=torch.float32)).item()}")
    print(f"Ones: {torch.mean(torch.tensor(set_iter_2['ones'], dtype=torch.float32)).item()} +/- {torch.std(torch.tensor(set_iter_2['ones'], dtype=torch.float32)).item()}")
    print(f"Predicted: {torch.mean(torch.tensor(set_iter_3['predicted'], dtype=torch.float32)).item()} +/- {torch.std(torch.tensor(set_iter_3['predicted'], dtype=torch.float32)).item()}")
    print(f"Ones: {torch.mean(torch.tensor(set_iter_3['ones'], dtype=torch.float32)).item()} +/- {torch.std(torch.tensor(set_iter_3['ones'], dtype=torch.float32)).item()}")


def dim(name):
    if name == 'mnist' or name == 'cifar' or name == 'mnist-cifar':
        return '$_{(28x28)}$'
    else:
        return '$_{(64x64)}$'
    
def plot_time(
        set_time,
        model_name
    ):
    
    def distance_over_time(
        set_time
    ):
        set_times = set_time[0]
        set_distance = set_time[1]
        time = [0]
        distance = []
        for i in range(len(set_times)):
            time.append(set_times[i] + time[-1])
            distance.append(torch.mean(set_distance[i]).cpu())
                    
        return time[1:], distance 
    
    fig, axs = plt.subplots(2, 3, figsize=(8, 5) ,sharex=False, sharey=False)
    axs = axs.flatten()
    lines = []
    labels = []
    for i, key in enumerate(set_time.keys()):
        time_bucket_pred_1, distance_pred_1 = distance_over_time(set_time[key]['predicted'])
        time_bucket_ones_1, distance_ones_1 = distance_over_time(set_time[key]['ones'])
        time_bucket_gauss_1, distance_gauss_1 = distance_over_time(set_time[key]['gauss'])
        # Plot the distance over time
       
        l1 = axs[i].plot(time_bucket_pred_1, distance_pred_1, label='UNOT', color='crimson')
        l3 = axs[i].plot(time_bucket_ones_1, distance_ones_1, label='Ones', color='indigo')
        l2 = axs[i].plot(time_bucket_gauss_1, distance_gauss_1, label='Gauss', color='cornflowerblue')

        if i == 0:  # Only store legend items once
            lines.extend([l1, l2, l3])
            labels.extend(['UNOT', 'Ones', 'Gauss'])
        
        axs[i].set_title(f'{key.upper()}{dim(key)}')
        if i % 3 == 0:
            axs[i].set_ylabel('Relative Error', fontsize=15)
        if i >= len(set_time.keys()) - 3:
            axs[i].set_xlabel('Time [s]', fontsize=15)
        axs[i].grid(False)
        if i <3:
            axs[i].set_xlim(right=0.005)
        else: 
            axs[i].set_xlim(right=0.02)
        axs[i].set_xlim(left=0)

    fig.legend(lines, labels, 
              loc='center',  # Center alignment
              bbox_to_anchor=(0.5, 0.025),  # Position below plots
              ncol=3,  # Display in 3 columns
              frameon=False)  # No frame around legend
    
    
    plt.subplots_adjust(wspace=0.2, hspace=0.2, bottom=0.15) 

    plt.tight_layout()
    plt.savefig(f'Images/time_relative_error_{model_name}.pdf')
    plt.show()

   
def plot_error_over_iter(
        set_error, 
        model_name,
    ):
    fig, axs = plt.subplots(2, 3, figsize=(8, 5),sharex=True, sharey=False)
    axs = axs.flatten()
    lines = []
    labels = []
    for i, key in enumerate(set_error.keys()):
        iter_pred = torch.stack(set_error[key]['predicted'][0]).mean(1).cpu()
        iter_ones = torch.stack(set_error[key]['ones'][0]).mean(1).cpu()
        iter_gauss = torch.stack(set_error[key]['gauss'][0]).mean(1).cpu()
        iter = torch.arange(0, 50)
        print(key, iter_pred[0])
        l1 = axs[i].plot(iter, iter_pred, color='crimson')[0]
        l2 = axs[i].plot(iter, iter_gauss, color='cornflowerblue')[0]
        l3 = axs[i].plot(iter, iter_ones, color='indigo')[0]
        axs[i].set_xticks(np.arange(0, 51, 10))

        if i == 0:  # Only store legend items once
            lines.extend([l1, l2, l3])
            labels.extend(['UNOT', 'Ones', 'Gauss'])
        
        axs[i].set_title(f'{key.upper()}{dim(key)}', fontsize=15)
        if i ==0 or i==3:
            axs[i].set_ylabel('Relative Error', fontsize=15)
        if i >= len(set_error.keys()) - 3:
            axs[i].set_xlabel('Iteration', fontsize=15)
        axs[i].grid(False)
        axs[i].set_ylim(top=0.3)

    #plt.tight_layout()
    fig.legend(lines, labels, 
              loc='center',  # Center alignment
              bbox_to_anchor=(0.5, 0.025),  # Position below plots
              ncol=3,  # Display in 3 columns
              frameon=False)  # No frame around legend
    
    plt.subplots_adjust(wspace=0.2, hspace=0.2, bottom=0.15) 
    plt.savefig(f'Images/iter_relative_error_{model_name}.pdf')
    plt.show()
        
    fig, axs = plt.subplots(2, 3, figsize=(8 , 5),sharex=True, sharey=False)
    axs = axs.flatten()
    lines = []
    labels = []
    for i, key in enumerate(set_error.keys()):
        iter_pred_mcv = torch.stack(set_error[key]['predicted'][1]).mean(1).cpu()
        iter_ones_mcv = torch.stack(set_error[key]['ones'][1]).mean(1).cpu()
        iter_gauss_mcv = torch.stack(set_error[key]['gauss'][1]).mean(1).cpu()

        iter = torch.arange(0, 50)
       

        l1 = axs[i].plot(iter, iter_pred_mcv, color='crimson')[0]
        l3 = axs[i].plot(iter, iter_ones_mcv, color='indigo')[0]
        l2 = axs[i].plot(iter, iter_gauss_mcv, color='cornflowerblue')[0]

        if i == 0:  # Only store legend items once
            lines.extend([l1, l2, l3])
            labels.extend(['UNOT', 'Ones', 'Gauss'])
        
        axs[i].set_title(f'{key.upper()}{dim(key)}', fontsize=15)
        #if i == 0 or i == 3:  # Nur für erste Spalte
        #    axs[i].set_ylabel('Marginal Constraint Violation')
        if i >= len(set_error.keys()) - 3:
            axs[i].set_xlabel('Iteration', fontsize=15)
        axs[i].grid(False)
        axs[i].set_ylim(top=0.3)
        axs[i].set_xticks(np.arange(0, 51, 10))
    fig.supylabel('Marginal Constraint Violation', x=0.06, fontsize=15)  
    #plt.tight_layout()
    fig.legend(lines, labels, 
              loc='center',  # Center alignment
              bbox_to_anchor=(0.5, 0.025),  # Position below plots
              ncol=3,  # Display in 3 columns
              frameon=False)  # No frame around legend
    
    
    plt.subplots_adjust(wspace=0.2, hspace=0.2, bottom=0.15) 
    plt.savefig(f'Images/iter_mcv_{model_name}.pdf')
    plt.show()


def plot_error_dim(set_dim, model_name):
    def color(key):
        if key == 'predicted':
            return 'crimson'
        elif key == 'ones':
            return 'indigo'
        else:
            return 'cornflowerblue'
        
    for key in set_dim.keys():
        plt.figure(figsize=(8, 6))
       
        for key2 in set_dim[key].keys():
            error = torch.stack(set_dim[key][key2]).cpu().T
            
            dim =torch.arange(10,64).repeat(error.size(0)) 
            print(dim.shape, error.flatten().shape)
            
            df_pred = pd.DataFrame({
                "Dimension": dim.numpy(),  # Convert to NumPy
                "Relative Error": error.flatten().numpy()
            })

            # Lineplot with mean and 95% confidence interval
            sns.lineplot(
                data=df_pred,
                x="Dimension",
                y="Relative Error",
                ci=95,  # 95% confidence interval
                estimator="mean",  # Use mean to aggregate data
                color=color(key2),  # Crimson line color
            )

        plt.title(f"{key} with 95% Confidence Interval", fontsize=14)
        plt.xlabel("Dimension", fontsize=12)
        plt.ylabel("Relative Error", fontsize=12)
        plt.grid(False, linestyle="--", alpha=0.7)
        plt.savefig(f'Images/dim_relative_error_{key}_{model_name}.pdf')
        plt.show()


def plot_error_dim_eps_matrix(data_set, set_error_dim, model_name):
    fig, axes = plt.subplots(len(data_set), len(data_set), figsize=(17, 15), sharex=True, sharey=True)
    #fig.suptitle("Relative Error across Dataset Pairs and Dimension and Regularization Parameter $\epsilon$", fontsize=27)

    for i, data1 in enumerate(data_set):
        for j, data2 in enumerate(data_set):
            if data1 =='chinese-mnist' or data2 == 'chinese-mnist' :
                continue
            ax = axes[i, j]
            if j <= i:
                xvals = torch.logspace(-2, 0, 40).numpy()

                error = set_error_dim[data1][data2]
                im = ax.imshow(error, cmap="hot_r",aspect='auto', vmin=0, vmax=0.15,extent=(xvals[0], xvals[-1], 10, 70), 
                    origin='lower'
                )

                ax.set_xscale('log')
                ax.set_xticks([1e-2, 1e-1, 1])
                ax.set_xticklabels([r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'], fontsize=16)

                ax.set_yticks(range(10, 75, 10))
                ax.set_yticklabels(range(10, 75, 10), fontsize=16)

            else:
                # Hide axes where j < i
                ax.axis('off')

            # Set subplot title
            if i == 0:  # Top row
                ax.set_title('EXPRESSIONS'.upper() if data2 == 'facialexpression' else data2.upper(), fontsize=22)
            if j == 0:  # Left column
                ax.set_ylabel('EXPRESSIONS'.upper() if data1 == 'facialexpression' else data1.upper(), fontsize=22)
            if i == 5:  # Left column
                ax.set_xlabel('$\epsilon$', fontsize=22)


    # Add a single colorbar to the side
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax.set_ylabel('Relative Error', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=20)

    # Adjust layout and show
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Leave space for the colorbar
    plt.savefig(f'Images/dim_eps_relative_error_matrix_{model_name}.pdf')
    plt.show()


def plot_error_dim_matrix(data_set, set_error_dim, model_name):
    def color(key):
        if key == 'predicted':
            return 'crimson'
        elif key == 'ones':
            return 'indigo'
        else:
            return 'cornflowerblue'
        # Create the grid plot
    fig, axes = plt.subplots(len(data_set), len(data_set), figsize=(17, 15), sharex=True, sharey=True)
    #fig.suptitle("Relative Error across Dataset Pairs and Dimension", fontsize=25)

    for i, data1 in enumerate(data_set):
        for j, data2 in enumerate(data_set):
            ax = axes[i, j]
            if j <= i:
                for key in set_error_dim[data1][data2].keys():
                    error = torch.stack(set_error_dim[data1][data2][key]).cpu().T
                    dim = torch.tensor(range(10, 64, 2)).repeat(error.size(0))
                    # Prepare DataFrame for Seaborn
                    df_pred = pd.DataFrame({
                        "Dimension": dim.numpy(),
                        "Relative Error": error.flatten().numpy()
                    })
                    
                    # Plot in the corresponding subplot
                    sns.lineplot(
                        data=df_pred,
                        x="Dimension",
                        y="Relative Error",
                        errorbar=('ci', 95),
                        estimator="mean",
                        ax=ax,
                        color=color(key),
                        label='UNOT' if key=='predicted' else ('Ones' if key=='ones' else 'Gauss')
                    )
                    ax.get_legend().remove()
                    ax.set_xlabel('', fontsize=22)  # Set the font size for x-axis label
                    
                    ax.set_yticks(np.arange(0, 0.9, 0.2))
                    ax.set_yticklabels([f'{x:.1f}' for x in np.arange(0, 0.9, 0.2)], fontsize=16)

                    ax.set_xticks(range(10, 64, 10))
                    ax.set_xticklabels(range(10, 64, 10), fontsize=16)
            else:
                # Hide axes where j < i
                ax.axis('off')
            # Set subplot title
            if i == 0:  # Top row
                ax.set_title('EXPRESSIONS'.upper() if data2 == 'facialexpression' else data2.upper(), fontsize=22)
            if j == 0:  # Left column
                ax.set_ylabel('EXPRESSIONS'.upper() if data1 == 'facialexpression' else data1.upper(), fontsize=22)
    handles, labels = axes[0,0].get_legend_handles_labels()
    handles = [
        Line2D([0], [0], color="crimson", linewidth=4, label="UNOT"),  # Thicker line
        Line2D([0], [0], color="indigo", linewidth=4, label="Ones"),
        Line2D([0], [0], color="cornflowerblue", linewidth=4, label="Gauss")
    ]
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, .90), ncol=3, fontsize=20)
    # Add a single xlabel and ylabel
    fig.text(0.5, 0.04, 'Resolution', ha='center', fontsize=25)
    fig.text(0.04, 0.5, 'Relative Error', va='center', rotation='vertical', fontsize=25)
    # Adjust layout and show
    plt.tight_layout(rect=[0.06, 0.06, 0.9, 0.95])  # Leave space for labels and colorbar
    plt.savefig(f'Images/dim_relative_error_matrix_{model_name}.pdf')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot the approximation evaluation.')
    parser.add_argument('--dimension', type=int, default=28, help='Dimension of the measures.') 
    parser.add_argument('--model', type=str, default='predictor_28_v2', help='Name of the model to use.')
    args = parser.parse_args()

    # load the data
    if args.dimension == 28:
        first_measure = 'mnist'
        second_measure = 'cifar'

    file_name_approx_mnist = f"experiments/approximation/one_step_{first_measure}_{first_measure}_{1000}_{args.model}.pt"
    file_name_approx_cifar = f"experiments/approximation/one_step_{second_measure}_{second_measure}_{1000}_{args.model}.pt"
    file_name_approx_mnist_cifar = f"experiments/approximation/one_step_{second_measure}_{first_measure}_{1000}_{args.model}.pt"

    file_name_iter_mnist = f"experiments/approximation/iteration_{first_measure}_{first_measure}_{1000}_{args.model}.pt"
    file_name_iter_cifar = f"experiments/approximation/iteration_{second_measure}_{second_measure}_{1000}_{args.model}.pt"
    file_name_iter_mnist_cifar = f"experiments/approximation/iteration_{second_measure}_{first_measure}_{1000}_{args.model}.pt"

    file_name_violation_mnist = f"experiments/approximation/violation_{first_measure}_{first_measure}_{1000}_{args.model}.pt"
    file_name_violation_cifar = f"experiments/approximation/violation_{second_measure}_{second_measure}_{1000}_{args.model}.pt"
    file_name_violation_mnist_cifar = f"experiments/approximation/violation_{second_measure}_{first_measure}_{1000}_{args.model}.pt"

    file_name_time_mnist = f"experiments/approximation/time_{first_measure}_{first_measure}_{1000}_{args.model}.pt"
    file_name_time_cifar = f"experiments/approximation/time_{second_measure}_{second_measure}_{1000}_{args.model}.pt"
    file_name_time_mnist_cifar = f"experiments/approximation/time_{second_measure}_{first_measure}_{1000}_{args.model}.pt"

    set_approximation_mnist = torch.load(file_name_approx_mnist)
    set_approximation_cifar = torch.load(file_name_approx_cifar)
    set_approximation_mnist_cifar = torch.load(file_name_approx_mnist_cifar)

    set_iteration_mnist = torch.load(file_name_iter_mnist)
    set_iteration_cifar = torch.load(file_name_iter_cifar)
    set_iteration_mnist_cifar = torch.load(file_name_iter_mnist_cifar)

    set_violation_mnist = torch.load(file_name_violation_mnist)
    set_violation_cifar = torch.load(file_name_violation_cifar)
    set_violation_mnist_cifar = torch.load(file_name_violation_mnist_cifar)

    set_time_mnist = torch.load(file_name_time_mnist)
    set_time_cifar = torch.load(file_name_time_cifar)
    set_time_mnist_cifar = torch.load(file_name_time_mnist_cifar)

    
    #plot_approximation(set_approximation_mnist, set_approximation_cifar, set_approximation_mnist_cifar, first_measure, second_measure)
    print_iteration(set_iteration_mnist, set_iteration_cifar, set_iteration_mnist_cifar)
    plot_approximation(set_violation_mnist, set_violation_cifar, set_violation_mnist_cifar, first_measure, second_measure)
    plot_time(set_time_mnist, set_time_cifar, set_time_mnist_cifar, first_measure, second_measure)


if __name__ == "__main__":
    main()