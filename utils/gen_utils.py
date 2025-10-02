import networkx as nx 
import numpy as np
import pandapower as pp
import warnings 
import torch
import copy
import os
import pickle
import pandas as pd 
from typing import Tuple, List, Dict, Union
from torch_geometric.data import Data, Batch, Dataset
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader 
from torch.utils.data import DataLoader as torch_loader
import torch.nn as nn
import matplotlib 
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pylab
import time 
import sys 

from scipy.stats import gaussian_kde
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon
from matplotlib.lines import Line2D

import logging 
import yaml
from datetime import datetime 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
import joblib 

from utils.ppnet_utils import use_stored_pfr, custom_se, check_pf_consistency, initialize_network

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

# Disable LaTeX rendering
plt.rcParams.update({
    'text.usetex': False,
    'mathtext.fontset': 'cm',
    'font.family': 'sans-serif'
})

###########################################################################################################

def save_model(trained_model: nn.Module, 
               path: str):
    torch.save(trained_model.state_dict(), path)



def precision_round(number, significant_digits=3):
    power = "{:e}".format(number).split('e')[1]
    return round(number, -(int(power) - significant_digits))



def create_report_directory(current_time: str) -> str:
    """Create and return the path to a new directory for report output."""
    # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = f"{parent_dir}/results/{current_time}/"
    os.makedirs(report_dir, exist_ok=True)
    return report_dir


def setup_plotting_style(usetex: bool = False):
    """Set up the plotting style for consistent visualization."""
    sns.set_theme(style="whitegrid", font_scale=1.3)
    plt.rcParams['text.usetex'] = True


def save_figure(fig, filename_base, report_dir):
    """Save a figure in both PDF and PNG formats."""
    try: 
        plt.savefig(f"{report_dir}/{filename_base}.pdf", dpi=300)
        plt.savefig(f"{report_dir}/{filename_base}.png", dpi=300)
    except Exception as e: 
        print(f"Saving pdf/png skipped!: {e}")
        


def plot_boxplot(data_df, title, xlabel, ylabel, xtick_rotation=90, figsize=(10, 6), fontsize: int = 20):
    """Create and return a boxplot figure."""
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    sns.boxplot(data_df, ax=ax)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.set_xticks(range(len(data_df.columns)))
    ax.set_xticklabels(data_df.columns + 1)
    plt.xticks(rotation=xtick_rotation)
    ax.tick_params(axis='both', labelsize=fontsize)
    return fig, ax


def generate_load_boxplots(sampled_input_data: Dict, report_dir: str):
    """Generate boxplots for load active and reactive power."""
    # LOAD active power
    load_p_df = pd.DataFrame(sampled_input_data['sim_load_p_mw'].T)
    fig_p, _ = plot_boxplot(
        load_p_df, 
        "Variability of load P (if present) at each bus",
        "Buses", 
        "MW"
    )
    save_figure(fig_p, "loadstd_box", report_dir)
    
    # LOAD reactive power
    load_q_df = pd.DataFrame(sampled_input_data['sim_load_q_mvar'].T)
    fig_q, _ = plot_boxplot(
        load_q_df, 
        "Variability of load Q (if present) at each bus",
        "Buses", 
        "MVAR"
    )
    save_figure(fig_q, "loadstdq_box", report_dir)


def generate_bus_boxplots(sampled_input_data: Dict, report_dir: str):
    """Generate boxplots for bus voltage, angle, active and reactive power."""
    vm_pu_df = pd.DataFrame(sampled_input_data['res_bus_vm_pu'].T)
    va_deg_df = pd.DataFrame(sampled_input_data['res_bus_va_deg'].T)
    p_mw_df = pd.DataFrame(sampled_input_data['res_bus_p_mw'].T)
    q_mvar_df = pd.DataFrame(sampled_input_data['res_bus_q_mvar'].T)

    # Voltage
    fig_v, _ = plot_boxplot(
        vm_pu_df, 
        "Variability of voltage at each bus",
        "Buses", 
        "Value"
    )
    save_figure(fig_v, "vmpu_box", report_dir)

    # Angle
    fig_a, _ = plot_boxplot(
        va_deg_df, 
        "Variability of angle at each bus",
        "Buses", 
        "Value"
    )
    save_figure(fig_a, "adeg_box", report_dir)

    # Active power
    fig_p, _ = plot_boxplot(
        p_mw_df, 
        "Variability of active power at each bus",
        "Buses", 
        "Value"
    )
    save_figure(fig_p, "pmw_box", report_dir)

    # Reactive power
    fig_q, _ = plot_boxplot(
        q_mvar_df, 
        "Variability of reactive power at each bus",
        "Buses", 
        "Value"
    )
    save_figure(fig_q, "qmvar_box", report_dir)


def plot_histogram(data, x_column, xlabel, ylabel, title, figsize=(10, 6), binwidth=None, fontsize: int = 20):
    """Create and return a histogram figure."""
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    if binwidth: 
        sns.histplot(data=data, x=x_column, ax=ax, binwidth=binwidth)
    else: 
        sns.histplot(data=data, x=x_column, ax=ax)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    return fig, ax


def generate_label_distributions(sampled_input_data: Dict, report_dir: str):
    """Generate distribution plots for voltage and angle labels."""
    # Voltage and angle (Unscaled)
    y_label_unscaled = inverse_scale(sampled_input_data['y_label'].reshape(-1,2), scaler=sampled_input_data['scaler_y_label'])
    y_label_unscaled_df = pd.DataFrame(y_label_unscaled.reshape(-1, 2))
    y_label_unscaled_df.columns = ['vm_pu (label)', 'va_deg (label)']
    
    # bin width for voltage histograms 
    # v_bin_width = 1e-4

    # Unscaled voltage
    fig_v, _ = plot_histogram(
        y_label_unscaled_df, 
        'vm_pu (label)',
        "Voltage Magnitude Label (pu)", 
        "Density",
        "Label V (pu) Distribution",
    )
    save_figure(fig_v, "vm_pu_label_unscaled", report_dir)
    
    # Unscaled angle
    fig_a, _ = plot_histogram(
        y_label_unscaled_df, 
        'va_deg (label)',
        "Voltage Angle Label (degree)", 
        "Density",
        "Label A (deg) Distribution"
    )
    save_figure(fig_a, "va_deg_label_unscaled", report_dir)
    
    # Voltage and angle (Scaled)
    y_label_df = pd.DataFrame(sampled_input_data['y_label'].reshape(-1, 2))
    y_label_df.columns = ['vm (label)', 'va (label)']
    
    # Scaled voltage
    fig_vs, _ = plot_histogram(
        y_label_df, 
        'vm (label)',
        "Voltage Magnitude Scaled (if present)", 
        "Density",
        "Label Distribution", 
    )
    save_figure(fig_vs, "vm_pu_label", report_dir)
    
    # Scaled angle
    fig_as, _ = plot_histogram(
        y_label_df, 
        'va (label)',
        "Voltage Angle (Standard Scaler)", 
        "Density",
        "Label Distribution"
    )
    save_figure(fig_as, "va_rad_label", report_dir)


def generate_branch_parameters_plot(sampled_input_data: Dict, report_dir: str):
    """Generate pairplots for branch parameters."""
    edge_input_features_df = pd.DataFrame(sampled_input_data['edge_input_feat'].reshape(-1, 6))
    
    # Line and transformer parameter distribution
    branch_param_df = edge_input_features_df[[1, 2, 3, 4]]
    branch_param_df.columns = ['r_pu', 'x_pu', 'g_pu', 'b_pu']
    
    
    
    # Save pairplot
    try: 
        # Creating pairplot
        pair_plot = sns.pairplot(branch_param_df, diag_kind='kde')
        plt.tight_layout()
        pair_plot.savefig(f"{report_dir}/param_joint_dist.pdf", dpi=300)
        pair_plot.savefig(f"{report_dir}/param_joint_dist.png", dpi=300)
    except Exception as e: 
        print(f"Saving pairplot failed! {e}")


def plot_loss_curves(all_losses: Tuple, config: Dict, report_dir: str, fontsize: int = 20):
    """Plot training and validation loss curves."""
    epochs = np.arange(config['training']['num_epochs'])
    train_loss_curve = all_losses[0]
    val_loss_curve = all_losses[1]

    # Identify best epoch (lowest validation loss)
    best_epoch = np.argmin(val_loss_curve) + 1
    best_val = val_loss_curve[best_epoch - 1]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=epochs, y=train_loss_curve, label='Training Loss', marker='o', ax=ax)
    sns.lineplot(x=epochs, y=val_loss_curve, label='Validation Loss', marker='o', ax=ax)
    
    # Highlight the best epoch
    plt.axvline(best_epoch, color='gray', linestyle='--', label=f"Best epoch ({best_epoch})")
    plt.scatter(best_epoch, best_val, color='blue', zorder=5)
    plt.text(best_epoch+0.5, best_val, f"{best_val:.4f}", color='red', verticalalignment="bottom")

    ax.set_xlabel('Epochs', fontsize=fontsize)
    ax.set_ylabel('Loss', fontsize=fontsize)
    ax.set_title('Training and Validation Loss', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    plt.legend()
    plt.tight_layout()
    
    save_figure(fig, "loss", report_dir)

def plot_gradient_norm_over_time(all_losses: Tuple, config: Dict, report_dir: str, fontsize: int = 20):
    """Plot gradient norm curve over training epochs."""
    epochs = np.arange(config['training']['num_epochs'])
    gradient_norms = all_losses[3]

    # Identify the epoch with the highest gradient norm (optional insight)
    max_grad_epoch = np.argmax(gradient_norms) + 1
    max_grad_val = gradient_norms[max_grad_epoch - 1]

    # Create plot 
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=epochs, y=gradient_norms, label='Gradient Norm', marker='o', ax=ax)

    # Highlight epoch with max gradient norm
    plt.axvline(max_grad_epoch, color='gray', linestyle='--', label=f"Max gradient ({max_grad_epoch})")
    plt.scatter(max_grad_epoch, max_grad_val, color='blue', zorder=5)
    plt.text(max_grad_epoch + 0.5, max_grad_val, f"{max_grad_val:.4f}", color='red', verticalalignment="bottom")

    ax.set_xlabel('Epochs', fontsize=fontsize)
    ax.set_ylabel('Gradient Norm', fontsize=fontsize)
    ax.set_title('Gradient Norm Over Time', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    plt.legend()
    plt.tight_layout()

    save_figure(fig, "gradient_norm", report_dir)


def plot_va_predictions(plot_loader: DataLoader, trained_model: nn.Module, 
                       sampled_input_data: Dict, report_dir: str = None,fontsize: int = 20):
    """Generate bar plots comparing predictions to labels for voltage and angle."""

    # Get prediction for a single graph
    single_graph = next(iter(plot_loader))
    with torch.no_grad():
        pred = trained_model(single_graph)
    
    if trained_model.name == "MultiTapSEGNN":
        pred_se, _ = pred
    else:
        pred_se = pred
    
    print("Plotting Voltage and Angles in PU and Degrees, respectively.\n\n")
    
    # Get scaled/unscaled predictions and labels
    is_scaler = sampled_input_data['scaler_node']
    if is_scaler:
        pred_se_va = inverse_scale(pred_se, scaler=sampled_input_data['scaler_y_label'])
        label_se_va = inverse_scale(single_graph[0].y, scaler=sampled_input_data['scaler_y_label'])
    else:
        pred_se_va = pred_se
        label_se_va = single_graph[0].y
    
    # Determine if we need to plot only first 100 nodes
    plot_first_100_nodes = pred_se_va.shape[0] >= 100


    # Get crest colors for consistent color coding
    # cp = "Paired"
    cp = "crest"
    crest_colors = sns.color_palette(cp, 2)
    # Create bar plots
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    width = 0.3
    
    if plot_first_100_nodes:
        v_min = torch.min(label_se_va[:100, 0])
        v_max = torch.max(label_se_va[:100, 0])
        xticks = np.arange(100)
        ax[0].bar(xticks - width/2, pred_se_va[:100, 0], width=width, label="Prediction", color=crest_colors[0])
        ax[0].bar(xticks + width/2, label_se_va[:100, 0], width=width, label="Label", color=crest_colors[1])
        ax[0].set_ylim(v_min-0.05, v_max+0.05)
        ax[1].bar(xticks - width/2, pred_se_va[:100, 1], width=width, label="Prediction", color=crest_colors[0])
        ax[1].bar(xticks + width/2, label_se_va[:100, 1], width=width, label="Label", color=crest_colors[1])
    else:
        v_min = torch.min(label_se_va[:, 0])
        v_max = torch.max(label_se_va[:, 0])
        xticks = np.arange(pred_se_va.shape[0])
        ax[0].bar(xticks - width/2, pred_se_va[:, 0], width=width, label="Prediction", color=crest_colors[0])
        ax[0].bar(xticks + width/2, label_se_va[:, 0], width=width, label="Label", color=crest_colors[1])
        ax[0].set_ylim(v_min-0.05, v_max+0.05)
        ax[1].bar(xticks - width/2, pred_se_va[:, 1], width=width, label="Prediction", color=crest_colors[0])
        ax[1].bar(xticks + width/2, label_se_va[:, 1], width=width, label="Label", color=crest_colors[1])
    

    ax[0].set_ylabel("Per Unit", fontsize=fontsize)
    ax[1].set_ylabel("Degree", fontsize=fontsize)
    ax[0].set_xlabel("Buses", fontsize=fontsize)
    ax[0].legend(loc='upper right', bbox_to_anchor=(1, 0.95), borderaxespad=0., fontsize=fontsize)
    ax[1].set_xlabel("Buses", fontsize=fontsize)
    
    # Update tick font sizes
    ax[0].tick_params(axis='both', labelsize=fontsize)
    ax[1].tick_params(axis='both', labelsize=fontsize)
    ax[1].legend(loc='best', fontsize=fontsize)
    # fig.suptitle(f"SE Predictions vs. Labels")
    fig.tight_layout()

    save_figure(fig, "va_barplot", report_dir)
    
    # Plot joint distribution predictions vs labels
    fig_j, ax_j = plt.subplots(2, 1, figsize=(10, 10))
    
    # Voltage predictions vs labels
    v_pred_label_df = pd.DataFrame(torch.vstack([pred_se_va[:, 0], label_se_va[:, 0]]).T,
                                columns=['Prediction', 'Label'])
    sns.kdeplot(v_pred_label_df, palette=cp, alpha=0.5, ax=ax_j[0], fill=True)
    ax_j[0].set_xlabel("Voltage Magnitude", fontsize=fontsize)
    ax_j[0].set_ylabel("KDE Density", fontsize=fontsize)
    
    # Angle predictions vs labels
    a_pred_label_df = pd.DataFrame(torch.vstack([pred_se_va[:, 1], label_se_va[:, 1]]).T,
                                columns=['Prediction', 'Label'])
    sns.kdeplot(a_pred_label_df, palette=cp, alpha=0.5, ax=ax_j[1], fill=True)
    ax_j[1].set_xlabel("Angle Magnitude", fontsize=fontsize)
    ax_j[1].set_ylabel("KDE Density", fontsize=fontsize)

    # Update tick font sizes
    ax_j[0].tick_params(axis='both', labelsize=fontsize)
    ax_j[1].tick_params(axis='both', labelsize=fontsize)
    
    if ax_j[0].get_legend() is not None:
        for text in ax_j[0].get_legend().get_texts():
            text.set_fontsize(fontsize)

    if ax_j[1].get_legend() is not None:
        for text in ax_j[1].get_legend().get_texts():
            text.set_fontsize(fontsize)

    fig_j.tight_layout()
    
    save_figure(fig_j, "va_pred_label_joint", report_dir)
    


def write_markdown_report(config: Dict, results: Dict, all_losses: Tuple, report_dir: str, usetex: bool = True):
    """Write the complete markdown report."""
    with open(f"{report_dir}/report.md", "w") as f:
        f.write("# 📝 Report \n \n")

        # Configuration
        f.write("## ⚙️ Configuration \n\n")
        yaml_config = yaml.dump(config, default_flow_style=False)
        f.write("```yaml\n")
        f.write(yaml_config)
        f.write("```\n\n")

        if usetex: 
            # Load distribution
            f.write('## Load distribution to sample synthetic power flow results\n\n')
            f.write("![LoadP box](loadstd_box.png)\n\n")
            f.write("![LoadQ box](loadstdq_box.png)\n\n")

            # Power Flow Results Distribution
            f.write('## Power Flow Results Distribution\n\n')
            f.write(f'The plots below show the variability of all states sampled by adding standard deviation in load.\n\n')
            f.write("![V box](vmpu_box.png)\n\n")
            f.write("![A box](adeg_box.png)\n\n")
            f.write("![P box](pmw_box.png)\n\n")
            f.write("![Q box](qmvar_box.png)\n\n")

            # Labels distribution
            f.write('## 📊 Label Distribution \n \n')
            f.write('### Unscaled \n\n')
            f.write("![Voltage Magnitude Labelunscale](vm_pu_label_unscaled.png)\n\n")
            f.write("![Voltage Angle Labelunscale](va_deg_label_unscaled.png)\n\n")
            
            f.write('### Scaled (Input to the model) \n\n')
            f.write("![Voltage Magnitude Label](vm_pu_label.png)\n\n")
            f.write("![Voltage Angle Label](va_rad_label.png)\n\n")

            # Parameter Distribution
            f.write('## 📊 Parameter Distribution \n \n')
            f.write("![Line and Trafo Parameter Distribution](param_joint_dist.png)\n\n")

            # Loss curves
            f.write("## 📉 Loss curve \n \n")
            f.write(f"![Training Loss](loss.png)\n\n")

            # Gradient curves 
            f.write("## 📉 Loss curve \n \n") 
            f.write(f"![Gradient Curve](gradient_norm.png)\n\n")

            # Predictions vs Labels
            f.write("### 📊 Predictions vs. Labels Bar Plot \n\n")
            f.write("![Predictions vs Labels](va_barplot.png)\n\n")

            # Predictions vs Labels Joint Distribution
            f.write('### Predictions vs. Labels Joint Distribution \n\n')
            f.write("![Pred vs. Labels kde](va_pred_label_joint.png)")

        else: 
            # Results
            f.write("## 🔎 Results \n\n")
            for key, value in results.items():
                f.write(f"- **{key}**: `{value}`\n")
            f.write(f"\n Test Loss = {all_losses[2]}\n\n")


def generate_markdown_report_and_save_model(
                           current_time: str, 
                           all_losses: Tuple, 
                           config: Dict, 
                           results: Dict,
                           plot_loader: DataLoader, 
                           trained_model: nn.Module,
                           sampled_input_data: Dict, 
                           usetex: bool = False, 
                           save_model_bool: bool = False):
    """
    Generate a professional report with visualizations of model performance and data distributions.
    """
    # Create directory for report files
    report_dir = create_report_directory(current_time=current_time)

    if save_model_bool == True:
        save_model(trained_model, path=f"{report_dir}/TapSEGNN.pth")
        print("Model saved!")

    
    # Set up plotting style
    setup_plotting_style(usetex=usetex)
    try: 
        # Generate all visualizations
        generate_load_boxplots(sampled_input_data, report_dir)
        generate_bus_boxplots(sampled_input_data, report_dir)
        generate_label_distributions(sampled_input_data, report_dir)
        generate_branch_parameters_plot(sampled_input_data, report_dir)
        plot_loss_curves(all_losses, config, report_dir)
        plot_gradient_norm_over_time(all_losses, config, report_dir)
        plot_va_predictions(plot_loader, trained_model, sampled_input_data, report_dir)
    except Exception as e: 
        print(f"Saving figures in the report skipped!: {e}")

    # Write the markdown report
    write_markdown_report(config, results, all_losses, report_dir, usetex=usetex)
    print(f"Report generated successfully in {report_dir}")
    

#####################################################################################
def generate_markdown_report_GAN_and_save_model(yaml_config: Dict, 
                                 train_g_losses: List, 
                                 train_d_losses: List, 
                                 train_d_accuracies: List, 
                                 parent_dir: str, 
                                 test_loader_G: DataLoader, 
                                 model_G: nn.Module, 
                                 sampled_input_data_G: Dict, 
                                 return_data: bool = False,
                                 seed: int = 0,
                                 usetex: bool = False,
                                 save_model_bool: bool = False):
    """
    Generate a comprehensive markdown report for GAN training results.
    
    Args:
        config (dict): Configuration dictionary to be saved as YAML
        train_d_losses (list): Discriminator training losses
        train_g_losses (list): Generator training losses  
        traind_d_accuracies (list): Discriminator training accuracies
        parent_dir (str): Parent directory for saving results
        additional_metrics (dict, optional): Additional metrics to include
        model_info (dict, optional): Model architecture information
    
    Returns:
        str: Path to the generated report directory
    """ 

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = f"{parent_dir}/results/GAN_only/{current_time}_{seed}"

    os.makedirs(report_dir, exist_ok=True)

    # save model 
    if save_model_bool == True:
        save_model(model_G, path=f"{report_dir}/TapSEGNN.pth")
        print("Generator Model saved!")
    # torch.save(model_G.state_dict(), report_dir)

    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    fontsize = 20
    plt.rcParams.update({
        'font.size': fontsize,
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'axes.linewidth': 1.2,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.5,
        'lines.markersize': 6,
    })

    colors = {
        'discriminator': '#1f77b4',  # Blue
        'generator': '#ff7f0e',      # Orange
        'accuracy': '#2ca02c'        # Green
    }
    # Generate training dynamics plot
    fig, ax1 = plt.subplots(figsize=(12, 8), dpi=300)
    epochs = range(len(train_g_losses))
    
    # Primary y-axis (losses)
    line1 = ax1.plot(epochs, train_d_losses,
                     color=colors['discriminator'],
                     linewidth=2.5,
                     label='Discriminator Loss',
                     marker='o',
                     markersize=4,
                     markevery=max(1, len(epochs)//40),
                     alpha=0.9)
    
    line2 = ax1.plot(epochs, train_g_losses,
                     color=colors['generator'],
                     linewidth=2.5,
                     label='Generator Loss',
                     marker='s',
                     markersize=4,
                     markevery=max(1, len(epochs)//40),
                     alpha=0.9)
    
    # Secondary y-axis (accuracy)
    ax2 = ax1.twinx()
    line3 = ax2.plot(epochs, train_d_accuracies,
                     color=colors['accuracy'],
                     linewidth=2.5,
                     label='Discriminator Accuracy',
                     marker='^',
                     markersize=4,
                     markevery=4,
                     alpha=0.9)
    
    # Axis formatting
    ax1.set_xlabel('Training Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax2.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_xlim(-0.5, len(epochs)-0.5)
    ax1.set_ylim(0, max(max(train_d_losses), max(train_g_losses)) * 1.1)
    ax2.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax2.grid(False)
    ax2.tick_params(axis='y', labelcolor=colors['accuracy'])
    ax2.spines['right'].set_color(colors['accuracy'])
    
    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc='upper center')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save training dynamics plot
    try: 
        plt.savefig(f"{report_dir}/FIG_training_dynamics.pdf",
                    dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.savefig(f"{report_dir}/FIG_training_dynamics.png",
                    dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    except Exception as e: 
        print(f"Saving training dynamics plot failed! {e}")
    plt.close()

    # Generate loss difference plot (convergence analysis)
    fig_cp, ax_cp = plt.subplots(1, 1, figsize=(12, 8), dpi=300)
    loss_diff = np.array(train_d_losses) - np.array(train_g_losses)
    
    ax_cp.plot(epochs, loss_diff, color='purple', linewidth=2, 
               label='Loss Difference (D - G)')
    ax_cp.axhline(y=0, color='red', linestyle='--', alpha=0.7, 
                  label='Nash Equilibrium')
    ax_cp.fill_between(epochs, loss_diff, alpha=0.3, color='purple')
    ax_cp.set_xlabel('Training Epoch', fontweight='bold')
    ax_cp.set_ylabel('Loss Difference', fontweight='bold')
    ax_cp.legend()
    ax_cp.grid(True, alpha=0.3)
    
    plt.tight_layout()
    try: 
        plt.savefig(f"{report_dir}/FIG_loss_difference.pdf",
                    dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.savefig(f"{report_dir}/FIG_loss_difference.png",
                    dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    except Exception as e: 
        print(f"Saving loss difference plots failed! {e}")
    plt.close()
    

    try: 
        batch_G = next(iter(test_loader_G))
        scaled_real_pv = batch_G[0].x_pfr
        scaled_real_p_edge = batch_G[1].x_pfr[:,0]
        scaled_gen_pv, scaled_gen_p_edge = model_G(batch_G)
        print("Forward pass calculated!")
    except Exception as e: 
        print(f"Exception occured at forward pass: {e}")
    
    try: 
        real_pv = inverse_scale(scaled_real_pv, scaler=sampled_input_data_G['scaler_node'])
        # all_real_p_values = real_pv[:,1].detach().cpu().numpy()
        # all_real_v_values = real_pv[:,0].detach().cpu().numpy()

        gen_pv = inverse_scale(scaled_gen_pv, scaler=sampled_input_data_G['scaler_node'])
        # all_generated_p_values = gen_pv[:,1].detach().cpu().numpy()
        # all_generated_v_values = gen_pv[:,0].detach().cpu().numpy()
        print("Inverse scale done!")
    except Exception as e: 
        print(f"Exception occured at calculating inverse: {e}")

    try:     
        p_edge_mean = sampled_input_data_G['scaler_edge'].mean_[0]
        p_edge_var = sampled_input_data_G['scaler_edge'].var_[0]

        real_p_edge = scaled_real_p_edge * p_edge_var + p_edge_mean
        # all_real_pedge_values = real_p_edge.detach().cpu().numpy()
        
        gen_p_edge = scaled_gen_p_edge * p_edge_var + p_edge_mean
        # all_generated_pedge_values = gen_p_edge.detach().cpu().numpy()

        # Convert tensors to numpy for plotting
        real_v_np = real_pv[:,0].detach().cpu().numpy()
        real_p_np = real_pv[:,1].detach().cpu().numpy()
        generated_v_np = gen_pv[:,0].detach().cpu().numpy()
        generated_p_np = gen_pv[:,1].detach().cpu().numpy()
        real_p_edge_np = real_p_edge.detach().cpu().numpy()
        gen_p_edge_np = gen_p_edge.detach().cpu().numpy()

        print(f"Mean and variances calculated!")
    except Exception as e: 
        print(f"Exception occured at calculating mean and variances: {e}")

    try: 
        net = initialize_network(yaml_config['data']['net_name'])
        plot_real_v_generated_linebar(report_dir=report_dir,
                                      net=net, 
                                      real_v=real_v_np, 
                                      real_p=real_p_np, 
                                      real_p_edge=real_p_edge_np, 
                                      gen_v=generated_v_np, 
                                      gen_p=generated_p_np, 
                                      gen_p_edge=gen_p_edge_np)
        print("Plotted real vs. generated line-bar plots!")
    except Exception as e: 
        print(f"Exception occured at plotting real vs. generated line-bar plots: {e}")
    
    try: 
        results_pf, simulated_pf_v = check_pf_consistency(net_name=yaml_config['data']['net_name'],
                                                            gen_v=generated_v_np, 
                                                            gen_p=generated_p_np, 
                                                        )
        fig, ax = plt.subplots(figsize=(12,8), constrained_layout=True)
        sns.set_theme(style="whitegrid", font_scale=1.3)
        if usetex == True: 
            plt.rcParams['text.usetex'] = True

        num_buses_range = np.arange(len(net.bus))
        num_buses = len(net.bus)
        plot_simgen_v = pd.DataFrame({
            'Bus Index': np.concatenate([num_buses_range, num_buses_range]),
            'Voltage (p.u.)': np.concatenate([simulated_pf_v, generated_v_np[:num_buses]]),
            'Type': ['Simulated PF'] * num_buses + ['Generated'] * num_buses
        })
        
        # try: 
        sns.lineplot(data=plot_simgen_v, x='Bus Index', y='Voltage (p.u.)', 
                hue='Type', marker='o', markersize=6, linewidth=2.5, ax=ax)
        # Customize the plot
        ax.set_xlabel('Sample Index', fontsize=14)
        ax.set_ylabel('Voltage Magnitude (p.u.)', fontsize=14)
        ax.set_title('Power Flow Consistency Check: Simulated vs Generated Voltages', fontsize=16, pad=20)
        ax.legend(title='Voltage Source', title_fontsize=12, fontsize=12)
        ax.grid(True, alpha=0.3)

        try: 
            plt.savefig(report_dir + f'/FIG_GAN_Simulated_vs_Generated_V.pdf', dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            plt.savefig(report_dir + f'/FIG_GAN_Simulated_vs_Generated_V.png', dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
        except Exception as e: 
            print(f"Saving GAN simulated vs. generated vm_pu failed! {e}")
        plt.close()
        
    except Exception as e: 
        print(f"At check_pf_consistency! {e}")

    
    # except Exception as e: 
    #     print(f"Physical Consistency failed because: {e}")

    
    try: 
        # Add statistical annotations
        real_power_mean = real_p_np.mean()
        real_voltage_mean = real_v_np.mean()
        gen_power_mean = generated_p_np.mean()
        gen_voltage_mean = generated_v_np.mean()


        # Create labeled DataFrames with better naming
        real_df = pd.DataFrame({
            'Power': real_p_np,
            'Voltage': real_v_np,
            'Type': 'Real Data'
        })

        gen_df = pd.DataFrame({
            'Power': generated_p_np,
            'Voltage': generated_v_np,
            'Type': 'Generated Data'
        })

        # Combine datasets
        combined_data = pd.concat([real_df, gen_df], ignore_index=True)
 
        # Set up the plot with enhanced styling
        plt.style.use('default')  # Keep default matplotlib style
        sns.set_palette(["#2E86AB", "#A23B72"])  # Professional blue and magenta

        # Create the enhanced jointplot
        g = sns.jointplot(
            data=combined_data,
            x='Power',
            y='Voltage',
            hue='Type',
            kind='hist',
            fill=True,
            alpha=0.6,
            height=10,
            ratio=5,
            space=0.1,
            fontsize=fontsize,
            marginal_kws={'alpha': 0.7, 'linewidth': 2},
            joint_kws={'alpha': 0.6, 'linewidths': 1.5},
            legend=True,  
        )

        # # Enhance the main plot
        g.ax_joint.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)


        # Enhance marginal plots
        g.ax_marg_x.grid(True, alpha=0.3)
        g.ax_marg_y.grid(True, alpha=0.3)

        # Add mean markers
        g.ax_joint.scatter(real_power_mean, real_voltage_mean, 
                        marker='x', s=200, color='#2E86AB', 
                        linewidth=3, label='Real Mean', zorder=5)
        g.ax_joint.scatter(gen_power_mean, gen_voltage_mean, 
                        marker='x', s=200, color='#A23B72', 
                        linewidth=3, label='Generated Mean', zorder=5)
        
        # Calculate and display statistical metrics
        def calculate_metrics(real_data, gen_data):
            """Calculate JS and KL divergence for Power and Voltage columns."""
            
            def compute_divergences(real_series, gen_series):
                # Combine data for common evaluation range
                data = np.concatenate([real_series, gen_series])
                x = np.linspace(data.min(), data.max(), 1000)
                
                # KDE estimation
                real_kde = gaussian_kde(real_series)
                gen_kde = gaussian_kde(gen_series)

                # Evaluate densities
                real_prob = real_kde(x)
                gen_prob = gen_kde(x)

                # Add epsilon to avoid zeros
                epsilon = 1e-10
                real_prob += epsilon
                gen_prob += epsilon

                # Re-normalize after epsilon adjustment
                real_prob /= np.sum(real_prob)
                gen_prob /= np.sum(gen_prob)

                # Compute divergences
                kl = np.sum(rel_entr(real_prob, gen_prob))  # KL(real || gen)
                js = jensenshannon(real_prob, gen_prob, base=2) ** 2  # Square to get divergence

                return js, kl

            # Compute metrics for both Power and Voltage
            js_div_power, kl_div_power = compute_divergences(real_data['Power'], gen_data['Power'])
            js_div_voltage, kl_div_voltage = compute_divergences(real_data['Voltage'], gen_data['Voltage'])

            return js_div_power, js_div_voltage, kl_div_power, kl_div_voltage

        js_div_power, js_div_voltage, kl_div_power, kl_div_voltage = calculate_metrics(real_df, gen_df)

        # Add text box with statistical information
        stats_text = f"""JS Divergence:
        Power: {js_div_power:.4f}
        Voltage: {js_div_voltage:.4f}

        KL Divergence:
        Power: {kl_div_power:.4f}
        Voltage: {kl_div_voltage:.4f}

        Sample Size:
        Real: {len(real_df)}
        Generated: {len(gen_df)}"""

        g.ax_joint.text(0.02, 0.98, stats_text,
                    transform=g.ax_joint.transAxes,
                    fontsize=16,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', 
                                facecolor='white', 
                                alpha=0.8,
                                edgecolor='gray'))

        # Get existing legend handles and labels from the seaborn plot
        handles, labels = g.ax_joint.get_legend_handles_labels()

        mean_handles = [
            Line2D([0], [0], marker='o', color='#2E86AB', linewidth=0, markersize=10, markeredgewidth=3),
            Line2D([0], [0], marker='o', color='#A23B72', linewidth=0, markersize=10, markeredgewidth=3)
        ]

        # Combine all handles and labels
        all_handles = handles + mean_handles
        all_labels = labels + ['Real Distribution', 'Generated Distribution']

        # Create the complete legend
        g.ax_joint.legend(handles=all_handles, 
                        labels=all_labels,
                        loc='lower left', 
                        fontsize=16)

        # Enhance legend with increased font size
        # g.ax_joint.legend(loc='lower right', 
        #                  fontsize=14,  # Use your fontsize variable
        #                 #  frameon=True, 
        #                 #  fancybox=True, 
        #                 #  shadow=True,
        #                 #  framealpha=0.9
        #                  )

        g.ax_joint.set_xlabel("$P$",fontsize=fontsize)
        g.ax_joint.set_ylabel("$|V\u0332|$",fontsize=fontsize)
        g.ax_joint.tick_params(axis='x', labelsize=fontsize)
        g.ax_joint.tick_params(axis='y', labelsize=fontsize)

        # Improve layout
        plt.tight_layout()

        # Save in multiple formats
        # plt.savefig(report_dir + f'/FIG_GAN_power_voltage_kde_comparison.pdf', dpi=300, bbox_inches='tight',
        #             facecolor='white', edgecolor='none')
        # plt.savefig(report_dir + f'/FIG_ch_results_GAN_power_voltage_kde_comparison.png', dpi=300, bbox_inches='tight',
        #             facecolor='white', edgecolor='none')
        plt.savefig(report_dir + f'/FIG_GAN_power_voltage_hist_comparison.pdf', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.savefig(report_dir + f'/FIG_ch_results_GAN_power_voltage_hist_comparison.png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        
        # Calculate and display statistical metrics
        def calculate_metrics(real_data, gen_data):
            """Calculate JS and KL divergence for Power and Voltage columns."""
            def compute_divergences(real_series, gen_series):
                # Combine data for common evaluation range
                data = np.concatenate([real_series, gen_series])
                x = np.linspace(data.min(), data.max(), 1000)
                
                # KDE estimation
                real_kde = gaussian_kde(real_series)
                gen_kde = gaussian_kde(gen_series)
                
                # Evaluate densities
                real_prob = real_kde(x)
                gen_prob = gen_kde(x)
                
                # Add epsilon to avoid zeros
                epsilon = 1e-10
                real_prob += epsilon
                gen_prob += epsilon
                
                # Re-normalize after epsilon adjustment
                real_prob /= np.sum(real_prob)
                gen_prob /= np.sum(gen_prob)
                
                # Compute divergences
                kl = np.sum(rel_entr(real_prob, gen_prob))  # KL(real || gen)
                js = jensenshannon(real_prob, gen_prob, base=2) ** 2  # Square to get divergence
                
                return js, kl
            
            # For edge power, we need to handle single series
            if isinstance(real_data, (pd.Series, np.ndarray)) and isinstance(gen_data, (pd.Series, np.ndarray)):
                js_div, kl_div = compute_divergences(real_data, gen_data)
                return js_div, kl_div
            else:
                # Compute metrics for both Power and Voltage
                js_div_power, kl_div_power = compute_divergences(real_data['Power'], gen_data['Power'])
                js_div_voltage, kl_div_voltage = compute_divergences(real_data['Voltage'], gen_data['Voltage'])
                return js_div_power, js_div_voltage, kl_div_power, kl_div_voltage

        # Create a DataFrame for plotting
        df_edge = pd.DataFrame({
            'Value': np.concatenate([real_p_edge_np, gen_p_edge_np]),
            'Type': ['Real Data'] * len(real_p_edge_np) + ['Generated Data'] * len(gen_p_edge_np)
        })

        # Calculate divergence metrics
        js_divergence, kl_divergence = calculate_metrics(real_p_edge_np, gen_p_edge_np)

        # Set up enhanced plotting style
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

        # Enhanced color palette
        colors = ['#2E86AB', '#A23B72']  # Professional blue and magenta
        sns.set_palette(colors)

        # Create the enhanced KDE plot
        hist_plot = sns.histplot(
            data=df_edge,
            x='Value',
            hue='Type',
            fill=True,
            common_norm=False,
            alpha=0.6,
            linewidth=2.5,
            ax=ax,
            # legend=True,
        )


        # kde_plot = sns.kdeplot(
        #     data=df_edge,
        #     x='Value',
        #     hue='Type',
        #     fill=True,
        #     common_norm=False,
        #     alpha=0.6,
        #     linewidth=2.5,
        #     ax=ax,
        #     # legend=True,
        # )

        # kde_plot.legend()

        # Add unfilled KDE lines for better definition
        # sns.kdeplot(
        #     data=df_edge,
        #     x='Value',
        #     hue='Type',
        #     fill=True,
        #     common_norm=False,
        #     linewidth=3,
        #     ax=ax,
        #     # legend=True,
        # )

        # Enhanced axis labels and title
        ax.set_xlabel('$P^+$', fontsize=fontsize)
        ax.set_ylabel('Histogram', fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)

        # ax.set_title('Distribution Comparison: Real vs Generated Edge Power',fontsize=fontsize, pad=20)

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)

        # Create comprehensive statistics text box
        stats_text = f"""Divergence Measures:
        JS Divergence: {js_divergence:.6f}
        KL Divergence: {kl_divergence:.6f}
        
        Sample Sizes:
        Real: {len(real_p_edge_np)}, Generated: {len(gen_p_edge_np)}"""

        # Add statistics text box
        ax.text(0.58, 0.98, stats_text,
                transform=ax.transAxes,
                fontsize=fontsize,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.8', 
                        facecolor='white', 
                        alpha=0.9,
                        edgecolor='gray',
                        linewidth=1))
        
        mean_handles = [
            Line2D([0], [0], marker='o', color='#2E86AB', linewidth=0, markersize=10, markeredgewidth=3),
            Line2D([0], [0], marker='o', color='#A23B72', linewidth=0, markersize=10, markeredgewidth=3)
        ]

        # Combine all handles and labels
        all_handles = mean_handles
        all_labels = ['Real Distribution', 'Generated Distribution']

        # Create the complete legend
        ax.legend(handles=all_handles, 
                        labels=all_labels,
                        loc='center right', 
                        fontsize=fontsize)

        # Enhance the plot area
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

        # Adjust layout
        plt.tight_layout()

        # Save in multiple formats
        # plt.savefig(report_dir + f'/FIG_GAN_edge_power_kde_enhanced.pdf', dpi=300, bbox_inches='tight',
        #             facecolor='white', edgecolor='none')
        # plt.savefig(report_dir + f'/FIG_GAN_edge_power_kde_enhanced.png', dpi=300, bbox_inches='tight',
        #             facecolor='white', edgecolor='none')
        plt.savefig(report_dir + f'/FIG_GAN_edge_power_hist_enhanced.pdf', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.savefig(report_dir + f'/FIG_GAN_edge_power_hist_enhanced.png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        
        plt.close()

    except Exception as e: 
        print(f"KDE plots failed!: {e}")
        kde_plot = False

    try: 
        # # Add statistical annotations
        # real_power_mean = real_p_np.mean()
        # real_voltage_mean = real_v_np.mean()
        # gen_power_mean = generated_p_np.mean()
        # gen_voltage_mean = generated_v_np.mean()

        # # Create labeled DataFrames with better naming
        # real_df = pd.DataFrame({
        #     'Power': real_p_np,
        #     'Voltage': real_v_np,
        #     'Type': 'Real Data'
        # })

        # gen_df = pd.DataFrame({
        #     'Power': generated_p_np,
        #     'Voltage': generated_v_np,
        #     'Type': 'Generated Data'
        # })

        # # Combine datasets
        # combined_data = pd.concat([real_df, gen_df], ignore_index=True)

        # # Set up the plot with enhanced styling
        # plt.style.use('default')  # Keep default matplotlib style
        # sns.set_palette(["#2E86AB", "#A23B72"])  # Professional blue and magenta

        # Create the enhanced jointplot with histograms
        g = sns.jointplot(
            data=combined_data,
            x='Power',
            y='Voltage',
            hue='Type',
            kind='hist',
            fill=True,
            alpha=0.6,
            height=10,
            ratio=5,
            space=0.1,
            # fontsize=fontsize,
            marginal_kws={'alpha': 0.7, 'linewidth': 2},
            joint_kws={'alpha': 0.6, 'linewidths': 1.5},
            legend=True,  
        )

        # Enhance the main plot
        g.ax_joint.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)

        # Enhance marginal plots
        g.ax_marg_x.grid(True, alpha=0.3)
        g.ax_marg_y.grid(True, alpha=0.3)

        # Add mean markers
        g.ax_joint.scatter(real_power_mean, real_voltage_mean, 
                        marker='x', s=200, color='#2E86AB', 
                        linewidth=3, label='Real Mean', zorder=5)
        g.ax_joint.scatter(gen_power_mean, gen_voltage_mean, 
                        marker='x', s=200, color='#A23B72', 
                        linewidth=3, label='Generated Mean', zorder=5)
        
        # Calculate and display statistical metrics using histogram-based comparison
        def calculate_metrics_histogram(real_data, gen_data):
            """Calculate JS and KL divergence for Power and Voltage columns using histograms."""
            
            def compute_divergences_histogram(real_series, gen_series, bins=50):
                # Determine common range for both datasets
                data_min = min(real_series.min(), gen_series.min())
                data_max = max(real_series.max(), gen_series.max())
                bin_edges = np.linspace(data_min, data_max, bins + 1)
                
                # Calculate histograms with same bins
                real_hist, _ = np.histogram(real_series, bins=bin_edges, density=True)
                gen_hist, _ = np.histogram(gen_series, bins=bin_edges, density=True)
                
                # Normalize to get probabilities
                real_prob = real_hist / np.sum(real_hist)
                gen_prob = gen_hist / np.sum(gen_hist)
                
                # Add epsilon to avoid zeros
                epsilon = 1e-10
                real_prob += epsilon
                gen_prob += epsilon
                
                # Re-normalize after epsilon adjustment
                real_prob /= np.sum(real_prob)
                gen_prob /= np.sum(gen_prob)

                # Compute divergences
                kl = np.sum(rel_entr(real_prob, gen_prob))  # KL(real || gen)
                js = jensenshannon(real_prob, gen_prob, base=2) ** 2  # Square to get divergence

                return js, kl

            # Compute metrics for both Power and Voltage
            js_div_power, kl_div_power = compute_divergences_histogram(real_data['Power'], gen_data['Power'])
            js_div_voltage, kl_div_voltage = compute_divergences_histogram(real_data['Voltage'], gen_data['Voltage'])

            return js_div_power, js_div_voltage, kl_div_power, kl_div_voltage

        js_div_power, js_div_voltage, kl_div_power, kl_div_voltage = calculate_metrics_histogram(real_df, gen_df)

        # Add text box with statistical information
        stats_text = f"""JS Divergence:
        Power: {js_div_power:.4f}
        Voltage: {js_div_voltage:.4f}

        KL Divergence:
        Power: {kl_div_power:.4f}
        Voltage: {kl_div_voltage:.4f}

        Sample Size:
        Real: {len(real_df)}
        Generated: {len(gen_df)}"""

        g.ax_joint.text(0.02, 0.98, stats_text,
                    transform=g.ax_joint.transAxes,
                    fontsize=16,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', 
                                facecolor='white', 
                                alpha=0.8,
                                edgecolor='gray'))

        # Get existing legend handles and labels from the seaborn plot
        handles, labels = g.ax_joint.get_legend_handles_labels()

        mean_handles = [
            Line2D([0], [0], marker='o', color='#2E86AB', linewidth=0, markersize=10, markeredgewidth=3),
            Line2D([0], [0], marker='o', color='#A23B72', linewidth=0, markersize=10, markeredgewidth=3)
        ]

        # Combine all handles and labels
        all_handles = handles + mean_handles
        all_labels = labels + ['Real Distribution', 'Generated Distribution']

        # Create the complete legend
        g.ax_joint.legend(handles=all_handles, 
                        labels=all_labels,
                        loc='lower left', 
                        fontsize=fontsize
                        )

        g.ax_joint.set_xlabel("$P$", fontsize=fontsize)
        g.ax_joint.set_ylabel("$|V\u0332|$", fontsize=fontsize)
        g.ax_joint.tick_params(axis='x', labelsize=fontsize)
        g.ax_joint.tick_params(axis='y', labelsize=fontsize)

        # Improve layout
        plt.tight_layout()

        # Save in multiple formats
        plt.savefig(report_dir + f'/FIG_GAN_power_voltage_hist_comparison.pdf', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.savefig(report_dir + f'/FIG_ch_results_GAN_power_voltage_hist_comparison.png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        
        # Calculate and display statistical metrics for edge power using histograms
        def calculate_metrics_histogram_single(real_data, gen_data, bins=50):
            """Calculate JS and KL divergence for single series using histograms."""
            def compute_divergences_histogram_single(real_series, gen_series, bins=50):
                # Determine common range for both datasets
                data_min = min(real_series.min(), gen_series.min())
                data_max = max(real_series.max(), gen_series.max())
                bin_edges = np.linspace(data_min, data_max, bins + 1)
                
                # Calculate histograms with same bins
                real_hist, _ = np.histogram(real_series, bins=bin_edges, density=True)
                gen_hist, _ = np.histogram(gen_series, bins=bin_edges, density=True)
                
                # Normalize to get probabilities
                real_prob = real_hist / np.sum(real_hist)
                gen_prob = gen_hist / np.sum(gen_hist)
                
                # Add epsilon to avoid zeros
                epsilon = 1e-10
                real_prob += epsilon
                gen_prob += epsilon
                
                # Re-normalize after epsilon adjustment
                real_prob /= np.sum(real_prob)
                gen_prob /= np.sum(gen_prob)
                
                # Compute divergences
                kl = np.sum(rel_entr(real_prob, gen_prob))  # KL(real || gen)
                js = jensenshannon(real_prob, gen_prob, base=2) ** 2  # Square to get divergence
                
                return js, kl
            
            # For edge power, we handle single series
            if isinstance(real_data, (pd.Series, np.ndarray)) and isinstance(gen_data, (pd.Series, np.ndarray)):
                js_div, kl_div = compute_divergences_histogram_single(real_data, gen_data, bins)
                return js_div, kl_div
            else:
                # Compute metrics for both Power and Voltage
                js_div_power, kl_div_power = compute_divergences_histogram_single(real_data['Power'], gen_data['Power'], bins)
                js_div_voltage, kl_div_voltage = compute_divergences_histogram_single(real_data['Voltage'], gen_data['Voltage'], bins)
                return js_div_power, js_div_voltage, kl_div_power, kl_div_voltage

        # Create a DataFrame for plotting edge power
        df_edge = pd.DataFrame({
            'Value': np.concatenate([real_p_edge_np, gen_p_edge_np]),
            'Type': ['Real Data'] * len(real_p_edge_np) + ['Generated Data'] * len(gen_p_edge_np)
        })

        # Calculate divergence metrics for edge power
        js_divergence, kl_divergence = calculate_metrics_histogram_single(real_p_edge_np, gen_p_edge_np)

        # Set up enhanced plotting style for edge power
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

        # Enhanced color palette
        colors = ['#2E86AB', '#A23B72']  # Professional blue and magenta
        sns.set_palette(colors)

        # Create the enhanced histogram plot for edge power
        hist_plot = sns.histplot(
            data=df_edge,
            x='Value',
            hue='Type',
            fill=True,
            alpha=0.6,
            bins=50,  # Specify number of bins
            stat='density',  # Use density for better comparison
            common_norm=False,
            linewidth=2,
            ax=ax,
        )

        # Enhanced axis labels
        ax.set_xlabel('$P^+$', fontsize=fontsize)
        ax.set_ylabel('Density', fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)

        # Add grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)

        # Create comprehensive statistics text box
        stats_text = f"""Divergence Measures:
        JS Divergence: {js_divergence:.6f}
        KL Divergence: {kl_divergence:.6f}
        
        Sample Sizes:
        Real: {len(real_p_edge_np)}, Generated: {len(gen_p_edge_np)}"""

        # Add statistics text box
        ax.text(0.58, 0.98, stats_text,
                transform=ax.transAxes,
                fontsize=fontsize,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.8', 
                        facecolor='white', 
                        alpha=0.9,
                        edgecolor='gray',
                        linewidth=1))
        
        # Create custom legend handles
        mean_handles = [
            Line2D([0], [0], marker='s', color='#2E86AB', linewidth=0, markersize=10, markeredgewidth=0),
            Line2D([0], [0], marker='s', color='#A23B72', linewidth=0, markersize=10, markeredgewidth=0)
        ]

        # Combine all handles and labels
        all_handles = mean_handles
        all_labels = ['Real Distribution', 'Generated Distribution']

        # Create the complete legend
        ax.legend(handles=all_handles, 
                        labels=all_labels,
                        loc='center right', 
                        fontsize=fontsize)

        # Enhance the plot area
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

        # Adjust layout
        plt.tight_layout()

        # Save in multiple formats
        plt.savefig(report_dir + f'/FIG_GAN_edge_power_hist_enhanced.pdf', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.savefig(report_dir + f'/FIG_GAN_edge_power_hist_enhanced.png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        
    except Exception as e: 
        print(f"Report generation failed at histogram plots: {e}")
        kde_plot = False


    # Generate markdown report
    with open(f"{report_dir}/training_report.md", 'w') as f:
        f.write("# 📝 GAN Training Report\n\n")
        f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Configuration section
        f.write("## ⚙️ Configuration\n\n")
        yaml_config = yaml.dump(yaml_config, default_flow_style=False)
        f.write("```yaml\n")
        f.write(yaml_config)
        f.write("```\n\n")
        
        # Training Summary
        f.write("## 📊 Training Summary\n\n")
        f.write(f"- **Total Epochs:** {len(train_g_losses)}\n")
        f.write(f"- **Final Generator Loss:** {train_g_losses[-1]:.4f}\n")
        f.write(f"- **Final Discriminator Loss:** {train_d_losses[-1]:.4f}\n")
        f.write(f"- **Final Discriminator Accuracy:** {train_d_accuracies[-1]:.4f}\n")
        f.write(f"- **Loss Difference (D-G):** {train_d_losses[-1] - train_g_losses[-1]:.4f}\n\n")
        
        
        # Training Dynamics
        f.write("## 📈 Training Dynamics\n\n")
        f.write("The following plot shows the evolution of generator and discriminator losses ")
        f.write("along with discriminator accuracy throughout training:\n\n")
        f.write("![Training Dynamics](FIG_training_dynamics.png)\n\n")
        
        # Convergence Analysis
        f.write("## 🎯 Convergence Analysis\n\n")
        f.write("The loss difference plot shows how close the GAN is to Nash equilibrium ")
        f.write("(where discriminator and generator losses are balanced):\n\n")
        f.write("![Loss Difference](FIG_loss_difference.png)\n\n")
        
        # Statistical Analysis
        f.write("## 📋 Statistical Analysis\n\n")
        f.write("### Loss Statistics\n\n")
        f.write("| Metric | Generator | Discriminator |\n")
        f.write("|--------|-----------|---------------|\n")
        f.write(f"| Mean | {np.mean(train_g_losses):.4f} | {np.mean(train_d_losses):.4f} |\n")
        f.write(f"| Std | {np.std(train_g_losses):.4f} | {np.std(train_d_losses):.4f} |\n")
        f.write(f"| Min | {np.min(train_g_losses):.4f} | {np.min(train_d_losses):.4f} |\n")
        f.write(f"| Max | {np.max(train_g_losses):.4f} | {np.max(train_d_losses):.4f} |\n\n")
        
        f.write("### Discriminator Accuracy Statistics\n\n")
        f.write(f"- **Mean Accuracy:** {np.mean(train_d_accuracies):.4f}\n")
        f.write(f"- **Standard Deviation:** {np.std(train_d_accuracies):.4f}\n")
        f.write(f"- **Min Accuracy:** {np.min(train_d_accuracies):.4f}\n")
        f.write(f"- **Max Accuracy:** {np.max(train_d_accuracies):.4f}\n\n")
        
        # Training Insights
        f.write("## 💡 Training Insights\n\n")
        
        # Convergence assessment
        final_loss_diff = abs(train_d_losses[-1] - train_g_losses[-1])
        if final_loss_diff < 0.1:
            f.write("✅ **Good Convergence**: The generator and discriminator losses are well balanced.\n\n")
        elif final_loss_diff < 0.5:
            f.write("⚠️ **Moderate Convergence**: Some imbalance between generator and discriminator.\n\n")
        else:
            f.write("❌ **Poor Convergence**: Significant imbalance between generator and discriminator losses.\n\n")
        
        # Discriminator accuracy assessment
        mean_acc = np.mean(train_d_accuracies)
        if 0.4 <= mean_acc <= 0.6:
            f.write("✅ **Optimal Discriminator Performance**: Accuracy around 50% indicates good balance.\n\n")
        elif mean_acc > 0.8:
            f.write("⚠️ **Discriminator Too Strong**: High accuracy may indicate generator struggling.\n\n")
        elif mean_acc < 0.3:
            f.write("⚠️ **Discriminator Too Weak**: Low accuracy may indicate poor discriminator training.\n\n")
        
        # Loss stability assessment
        g_loss_std = np.std(train_g_losses[-10:])  # Last 10 epochs
        d_loss_std = np.std(train_d_losses[-10:])
        if g_loss_std < 0.1 and d_loss_std < 0.1:
            f.write("✅ **Stable Training**: Low variance in recent losses indicates stability.\n\n")
        else:
            f.write("⚠️ **Training Instability**: High variance in recent losses detected.\n\n")
        
        # Statistical Summary
        f.write("## 📈 Statistical Summary\n\n")
        f.write("### Node Features (Power & Voltage)\n\n")
        f.write("| Metric | Real Data | Generated Data | Difference |\n")
        f.write("|--------|-----------|----------------|------------|\n")
        f.write(f"| Power Mean | {real_power_mean:.4f} | {gen_power_mean:.4f} | {abs(real_power_mean - gen_power_mean):.4f} |\n")
        f.write(f"| Voltage Mean | {real_voltage_mean:.4f} | {gen_voltage_mean:.4f} | {abs(real_voltage_mean - gen_voltage_mean):.4f} |\n")
        f.write(f"| Power Std | {real_p_np.std():.4f} | {generated_p_np.std():.4f} | {abs(real_p_np.std() - generated_p_np.std()):.4f} |\n")
        f.write(f"| Voltage Std | {real_v_np.std():.4f} | {generated_v_np.std():.4f} | {abs(real_v_np.std() - generated_v_np.std()):.4f} |\n\n")
        
        f.write("### Edge Features (Power Flow)\n\n")
        f.write("| Metric | Real Data | Generated Data | Difference |\n")
        f.write("|--------|-----------|----------------|------------|\n")
        f.write(f"| Edge Power Mean | {real_p_edge_np.mean():.4f} | {gen_p_edge_np.mean():.4f} | {abs(real_p_edge_np.mean() - gen_p_edge_np.mean()):.4f} |\n")
        f.write(f"| Edge Power Std | {real_p_edge_np.std():.4f} | {gen_p_edge_np.std():.4f} | {abs(real_p_edge_np.std() - gen_p_edge_np.std()):.4f} |\n\n")
        
        # Data Quality Assessment
        f.write("## 🔍 Data Quality Assessment\n\n")
        f.write("### Sample Sizes\n\n")
        f.write(f"- **Real Data Points:** {len(real_df)}\n")
        f.write(f"- **Generated Data Points:** {len(gen_df)}\n")
        f.write(f"- **Edge Data Points (Real):** {len(real_p_edge_np)}\n")
        f.write(f"- **Edge Data Points (Generated):** {len(gen_p_edge_np)}\n\n")

        f.write("### Key Findings\n\n")
        power_diff_pct = abs(real_power_mean - gen_power_mean) / abs(real_power_mean) * 100
        voltage_diff_pct = abs(real_voltage_mean - gen_voltage_mean) / abs(real_voltage_mean) * 100
        edge_diff_pct = abs(real_p_edge_np.mean() - gen_p_edge_np.mean()) / abs(real_p_edge_np.mean()) * 100
        
        f.write(f"- **Power Mean Deviation:** {power_diff_pct:.2f}%\n")
        f.write(f"- **Voltage Mean Deviation:** {voltage_diff_pct:.2f}%\n")
        f.write(f"- **Edge Power Mean Deviation:** {edge_diff_pct:.2f}%\n\n")

        
        # if kde_plot: 
            # Model Performance Metrics
        f.write("## 🎯 Model Performance Metrics\n\n")
        f.write("### Power-Voltage Distribution Analysis\n\n")
        f.write(f"- **JS Divergence (Power):** {js_div_power:.6f}\n")
        f.write(f"- **JS Divergence (Voltage):** {js_div_voltage:.6f}\n")
        f.write(f"- **KL Divergence (Power):** {kl_div_power:.6f}\n")
        f.write(f"- **KL Divergence (Voltage):** {kl_div_voltage:.6f}\n\n")
        
        f.write("### Edge Power Distribution Analysis\n\n")
        f.write(f"- **JS Divergence:** {js_divergence:.6f}\n")
        f.write(f"- **KL Divergence:** {kl_divergence:.6f}\n\n")
        
        # Generated Visualizations
        f.write("## 📊 Generated Visualizations\n\n")
        f.write("The following visualizations have been generated and saved:\n\n")

        try: 
            f.write("1. **Power-Voltage KDE Comparison**\n")
            f.write("![pv_compare](FIG_GAN_power_voltage_kde_comparison.png)\n")            
            
            f.write("2. **Edge Power KDE Distribution**\n")
            f.write("![edge_power](FIG_GAN_edge_power_kde_enhanced.png)\n")
            
            f.write("1. **Power-Voltage Histogram Comparison**\n")
            f.write("![pv_compare](FIG_ch_results_GAN_power_voltage_hist_comparison.png)\n")            
            
            f.write("2. **Edge Power Histogram Distribution**\n")
            f.write("![edge_power](FIG_GAN_edge_power_hist_enhanced.png)\n")

            f.write("2. **P Real vs. Generated line plots**")
            f.write("![rg_line](FIG_GAN_PVPedge_lineplot.png)\n")
        except Exception as e: 
            print(e)
        
        try: 
            # Physical consistency of Output 
            f.write("## Physical Consistency of Imputed Measurements")
            f.write(f"1. Power Flow Converges? {results_pf['convergence_status']}\n\n")
            f.write(f"2. **Simulated vs. Generated Voltage Profile**\n")
            f.write("![sim_gen_v](FIG_GAN_Simulated_vs_Generated_V.png)")
        except Exception as e: 
            print(f"At markdown writing: {e}")

        # else: 
            # f.write(f"KDE plots not generated because of {e}")
        
        # Footer
        f.write("---\n\n")
        f.write("*This report was automatically generated from GAN training results. ")
        f.write("All metrics and visualizations are based on the latest model checkpoint.*\n")
        print(f"📄 Training report saved to: {report_dir}/training_report.md")
    
    if return_data: 
        generated_data = {'gen_v': generated_v_np, 
                        'gen_p': generated_p_np, 
                        'gen_p_edge': gen_p_edge}
        
        return generated_data, simulated_pf_v
    else: 
        return None 
    # return report_dir


#####################################################################################
def plot_real_v_generated_linebar(report_dir: str, 
                                    net: pp.pandapowerNet, 
                                    real_v: np.ndarray, 
                                    real_p: np.ndarray, 
                                    real_p_edge: np.ndarray, 
                                    gen_v: np.ndarray, 
                                    gen_p: np.ndarray, 
                                    gen_p_edge: np.ndarray):

    num_buses = len(net.bus)
    num_lines = len(net.line)
    num_trafos = len(net.trafo)
    num_edges = num_lines + num_trafos

    # Set seaborn style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 12)

    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(18, 12))
    fig.suptitle(f'Real vs Generated Data Comparison (Buses: {num_buses})', 
                 fontsize=16, fontweight='bold')

    # 1. Line plot for V values
    ax1 = axes[0]
    x_indices = np.arange(num_buses)
    sns.lineplot(x=x_indices, y=real_v[:num_buses], label='Real V', ax=ax1, color='blue', alpha=0.7)
    sns.lineplot(x=x_indices, y=gen_v[:num_buses], label='Generated V', ax=ax1, color='red', alpha=0.7)

    # Adaptive range for V
    v_min = min(np.min(real_v[:num_buses]), np.min(gen_v[:num_buses]))
    v_max = max(np.max(real_v[:num_buses]), np.max(gen_v[:num_buses]))
    v_margin = (v_max - v_min) * 0.05
    ax1.set_ylim(v_min - v_margin, v_max + v_margin)
    # ax1.set_title('Voltage over buses')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Voltage (p.u.)')
    ax1.legend()

    # 2. Line plot for P values
    ax2 = axes[1]
    real_p = real_p[:num_buses]
    gen_p = gen_p[:num_buses]
    sns.lineplot(x=x_indices, y=real_p, label='Real P', ax=ax2, color='green', alpha=0.7)
    sns.lineplot(x=x_indices, y=gen_p, label='Generated P', ax=ax2, color='orange', alpha=0.7)

    # Adaptive range for P
    p_min = min(np.min(real_p), np.min(gen_p))
    p_max = max(np.max(real_p), np.max(gen_p))
    p_margin = (p_max - p_min) * 0.05
    ax2.set_ylim(p_min - p_margin, p_max + p_margin)
    # ax2.set_title('Power Values Over Time')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Power (MW)')
    ax2.legend()

    # 3. Line plot for P_edge values
    ax3 = axes[2]
    x_indices_edge = np.arange(num_edges)
    real_p_edge = real_p_edge[:num_edges]
    gen_p_edge = gen_p_edge[:num_edges]
    sns.lineplot(x=x_indices_edge, y=real_p_edge, label='Real P Edge', ax=ax3, color='purple', alpha=0.7)
    sns.lineplot(x=x_indices_edge, y=gen_p_edge, label='Generated P Edge', ax=ax3, color='brown', alpha=0.7)

    # Adaptive range for P_edge
    p_edge_min = min(np.min(real_p_edge), np.min(gen_p_edge))
    p_edge_max = max(np.max(real_p_edge), np.max(gen_p_edge))
    p_edge_margin = (p_edge_max - p_edge_min) * 0.05
    ax3.set_ylim(p_edge_min - p_edge_margin, p_edge_max + p_edge_margin)
    # ax3.set_title('Edge Power Values Over Time')
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Edge Power (MW)')
    ax3.legend()

    # Adjust layout
    plt.tight_layout()

    # Save the figure with high quality
    try: 
        plt.savefig(report_dir + f'/FIG_GAN_PVPedge_lineplot.png', dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.savefig(report_dir + f'/FIG_GAN_PVPedge_lineplot.pdf', bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
    except Exception as e: 
        print(f"Saving GAN PVEdge line plot failed! {e}")

    plt.close()
    
    return fig, axes


#####################################################################################


def scale_numeric_columns(input_tensor: torch.Tensor, 
                          categorical_cols: int = None):
    """
    Scales only numerical columns in a 3d tensors, keeping categorical columns unchanged. (for tap positions)
   """

    tensor_list = list(input_tensor)

    # Convert tensors to numpy arrays for scaling
    tensor_np_list = [tensor.numpy() for tensor in tensor_list]

    # Identify numerical columns (all columns except categorical ones)
    all_cols = range(tensor_np_list[0].shape[1])
    if categorical_cols is None:
        numerical_cols = all_cols  # If no categorical cols, scale everything
        categorical_cols = []  # Empty categorical list
    else:
        numerical_cols = list(set(all_cols) - set(categorical_cols))  # Exclude categorical cols

    # Fit StandardScaler only on numerical columns
    scaler = StandardScaler()
    flat_numerical_data = np.vstack([tensor[:, numerical_cols] for tensor in tensor_np_list])
    scaler.fit(flat_numerical_data)

    # Transform only numerical columns
    scaled_tensor_list = [
        torch.tensor(
            np.hstack((scaler.transform(tensor[:, numerical_cols]), tensor[:, categorical_cols])), 
            dtype=torch.float32
        ) 
        for tensor in tensor_np_list
    ]
    
    scaled_tensor = torch.stack(scaled_tensor_list) # 3D Tensor

    return scaled_tensor, scaler


#####################################################################################


def inverse_scale(scaled_tensor: torch.Tensor, 
                  scaler: StandardScaler, 
                  categorical_cols: int = None):
    """
    Reverts the scaling of numerical columns to get back the original values.

    """
    # Apply inverse transform only to numerical columns
    if len(scaled_tensor.shape) == 3: 
        # obtain the numerical cols 
        all_cols = range(scaled_tensor.shape[2])
        if categorical_cols is None:
            numerical_cols = all_cols  # If no categorical cols, scale everything
            categorical_cols = []  # Empty categorical list
        else:
            numerical_cols = list(set(all_cols) - set(categorical_cols))  # Exclude categorical cols

        
        scaled_tensor_list = list(scaled_tensor)

        # Convert tensors to numpy arrays for inverse transformation
        scaled_np_list = [tensor.numpy() for tensor in scaled_tensor_list]

        original_tensor_list = [
            torch.tensor(
                np.hstack((scaler.inverse_transform(tensor[:, numerical_cols]), tensor[:, categorical_cols])), 
                dtype=torch.float32
            )
            for tensor in scaled_np_list
        ]
        original_tensor = torch.stack(original_tensor_list)
        return original_tensor
    else: # 2d 
        # obtain the numerical cols 
        all_cols = range(scaled_tensor.shape[1])
        if categorical_cols is None:
            numerical_cols = all_cols  # If no categorical cols, scale everything
            categorical_cols = []  # Empty categorical list
        else:
            numerical_cols = list(set(all_cols) - set(categorical_cols))  # Exclude categorical cols

        scaled_np = scaled_tensor.cpu().detach().numpy() 
        original_tensor = torch.tensor(
            np.hstack((scaler.inverse_transform(scaled_np[:, numerical_cols]), scaled_np[:, categorical_cols])), 
                dtype=torch.float32
        )
        return original_tensor


 


def get_edge_index_lg(edge_index: torch.Tensor) -> Tuple[torch.Tensor]: 
    """Calculates the edge-index representing the linegraph laplacian."""
    G = nx.Graph()
    edge_index_np = edge_index.cpu().numpy() # [2, num_edges]
    edges = [(edge_index_np[0,i], edge_index_np[1,i]) for i in range(edge_index_np.shape[1])]
    G.add_edges_from(edges)

    # calculate the B_1 node-edge incidence matrix 
    B_1 = torch.tensor(nx.incidence_matrix(G, oriented=True).toarray(), dtype=torch.float32)

    # linegraph adjacency 
    A_lg = torch.abs(B_1.T @ B_1 - 2 * torch.eye(B_1.shape[1], dtype=torch.float32))

    # linegraph laplacian 
    L_lg = torch.diag(A_lg @ torch.ones(B_1.shape[1], dtype=torch.float32)) - A_lg

    # get row and col indices
    L_lg = L_lg.numpy() 
    row_lg, col_lg = np.nonzero(L_lg)

    # add edge weights 
    edge_weight_lg = []
    for row_id, col_id in zip(row_lg, col_lg):
        edge_weight_lg.append(L_lg[row_id, col_id])

    edge_weight_lg = torch.tensor(edge_weight_lg, dtype=torch.float32)
    if tensor_any_nan(edge_weight_lg)[0]:
        raise ValueError("NaN in Line-graph Laplacian!")


    # create equivalent edge indices 
    edge_index_lg = torch.tensor(np.vstack((row_lg, col_lg)), dtype=torch.long)

    return edge_index_lg, edge_weight_lg


def get_edge_index_lu(edge_index: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor]]
]:
    """
    Calculate the edge-index representing the lower and upper laplacian.
    
    Args:
        edge_index: Either a single torch.Tensor of shape [2, num_edges] or 
                   a Dict[str, torch.Tensor] where each value has shape [2, num_edges]
    
    Returns:
        If input is torch.Tensor:
            Tuple of (edge_index_l, edge_index_u, edge_weight_l, edge_weight_u)
        If input is Dict[str, torch.Tensor]:
            Tuple of (Dict[edge_index_l], Dict[edge_index_u], Dict[edge_weight_l], Dict[edge_weight_u])
    """
    
    def _process_single_edge_index(edge_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process a single edge_index tensor."""
        G = nx.Graph()
        edge_index_np = edge_idx.cpu().numpy()  # [2, num_edges]
        edges = [(edge_index_np[0, i], edge_index_np[1, i]) for i in range(edge_index_np.shape[1])]
        G.add_edges_from(edges)
        
        (L_l, L_u), *_ = construct_hodge_laplacian(G=G)
        row_l, col_l = np.nonzero(L_l)
        row_u, col_u = np.nonzero(L_u)
        
        # add edge-weights
        edge_weight_l, edge_weight_u = [], []
        for row_id, col_id in zip(row_l, col_l):
            edge_weight_l.append(L_l[row_id, col_id])
        edge_weight_l = torch.tensor(edge_weight_l, dtype=torch.float32)
        
        for row_id, col_id in zip(row_u, col_u):
            edge_weight_u.append(L_u[row_id, col_id])
        edge_weight_u = torch.tensor(edge_weight_u, dtype=torch.float32)
        
        if tensor_any_nan(edge_weight_l, edge_weight_u)[0]:
            raise ValueError("NaN in Hodge Laplacian!")
        
        # create equivalent edge indices
        edge_index_l = torch.tensor(np.vstack((row_l, col_l)), dtype=torch.long)
        edge_index_u = torch.tensor(np.vstack((row_u, col_u)), dtype=torch.long)
        
        return edge_index_l, edge_index_u, edge_weight_l, edge_weight_u
    
    # Handle single tensor case
    if isinstance(edge_index, torch.Tensor):
        print("\n get_edge_index_lu handling single tensor...\n")
        time.sleep(3)
        return _process_single_edge_index(edge_index)
    
    # Handle dictionary case
    elif isinstance(edge_index, dict):
        print("\n get_edge_index_lu handling dictionary of tensors...\n")
        time.sleep(3)
        edge_index_l_dict = {}
        edge_index_u_dict = {}
        edge_weight_l_dict = {}
        edge_weight_u_dict = {}
        
        for key, edge_idx in edge_index.items():
            edge_idx_l, edge_idx_u, edge_w_l, edge_w_u = _process_single_edge_index(edge_idx)
            
            edge_index_l_dict[key] = edge_idx_l
            edge_index_u_dict[key] = edge_idx_u
            edge_weight_l_dict[key] = edge_w_l
            edge_weight_u_dict[key] = edge_w_u
        
        return edge_index_l_dict, edge_index_u_dict, edge_weight_l_dict, edge_weight_u_dict
    
    else:
        raise TypeError(f"edge_index must be torch.Tensor or Dict[str, torch.Tensor], got {type(edge_index)}")



def get_adjacency(net):
    """
    Calculate adjacency matrix for the pandapower net. 

    Args:
        net (pandapower DataFrame)

    Returns: 
        Adjacency Matrix (S) : np.ndarray; Shape: [num_nodes, num_nodes] or [num_buses, num_buses]
    
    """
    
    S = np.zeros([len(net.bus.index), len(net.bus.index)])

    # idea 1: adjacency_matrix 
    # edge weights as sqrt(r^2 + x^2)
    # for i,j,r,x in zip(net.line.from_bus.values, net.line.to_bus.values, net.line.r_ohm_per_km, net.line.x_ohm_per_km):
    #     S[i,j] = S[j,i] = np.sqrt(r*r + x*x)
    # # normalize S 
    # S = S / np.max(S)

    # idea 2
    # buses connected by lines
    for i, j in zip(net.line.from_bus.values, net.line.to_bus.values):
        S[i,j] = S[j,i] = 1
    
    # buses connected by trafos 
    for i, j in zip(net.trafo.hv_bus.values, net.trafo.lv_bus.values):
        S[i,j] = S[j,i] = 1

    return S

#####################################################################################

# check if the net has triangles or 2-simplices 
def any_2simplices(is_in_model: bool = False, 
                   G: nx.Graph = None,
                   data: Data = None) -> bool: 
    """Returns boolean indicating whether the graph has any triangles or not."""
    if is_in_model:     
        G = nx.Graph()
        edge_index = data.edge_index.cpu().numpy()
        edges = [(edge_index[0, i], edge_index[1, i]) for i in range(edge_index.shape[1])]
        G.add_edges_from(edges)
        any_triangles = sum(nx.triangles(G).values()) != 0
        return any_triangles
    # else 
    any_triangles = sum(nx.triangles(G).values()) != 0 
    return any_triangles

#####################################################################################

def get_edge_index_from_ppnet(net):
    """
    This function creates an edge-index tensor ([2, num_edges]) from pandapower net,
    considering both lines (from_bus to to_bus) and transformers (hv_bus to lv_bus).

    Args:
        net: Pandapower Net

    Returns:
        torch.Tensor: Edge-index in COO format; Shape [2, num_edges]
    """
    edges = []
    
    # Add lines (from_bus to to_bus)
    for _, line in net.line.iterrows():
        edges.append((line.from_bus, line.to_bus))
    
    # Add transformers (hv_bus to lv_bus)
    for _, trafo in net.trafo.iterrows():
        edges.append((trafo.hv_bus, trafo.lv_bus))
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    if tensor_any_nan(edge_index)[0]: 
        print(f"Edge index tensor = {edge_index}")
        raise ValueError("Nan in get_edge_index_from_ppnet")
    return edge_index


#####################################################################################

def construct_hodge_laplacian(G: nx.Graph) -> Tuple[
                                  Tuple[np.array, np.array], 
                                  Tuple[np.array, np.array]]:
    """This function creates Hodge-Laplacian and lower and upper incidence matrix given a Undirected Graph 
    by adding random orientation, making it a directed graph."""
    
    # directed graph 
    G_dir = nx.DiGraph()
    G_dir.add_nodes_from(G.nodes())
    G_dir.add_edges_from(G.edges())

    # get the incidence matrices 
    B_1 = nx.incidence_matrix(G_dir, oriented=True).toarray()
    
    
    num_edges = G_dir.number_of_edges()
    num_triangles = sum(nx.triangles(nx.Graph(G_dir)).values()) // 3

    B_2 = np.zeros((num_edges, num_triangles))

    # dictionary of edge indices 
    edge_to_index = {edge:i for i, edge in enumerate(G_dir.edges())}

    triangle_idx = 0 
    # find all the triangles and fill the B_2 incidence matrix 
    for triangle in nx.enumerate_all_cliques(nx.Graph(G_dir)): # since enumerate* works for undirected only 
        if len(triangle) == 3:
            sorted_tri = sorted(triangle)
            cyclic_edges = [(sorted_tri[i], sorted_tri[(i+1) % 3]) for i in range(3)]

            for edge in cyclic_edges: 
                if edge in G_dir.edges(): 
                    B_2[edge_to_index[edge], triangle_idx] = 1
                elif (edge[1], edge[0]) in G_dir.edges(): 
                    B_2[edge_to_index[(edge[1], edge[0])], triangle_idx] = -1
            triangle_idx += 1 
    
    # check if boundary condition satisfied 
    bc = B_1 @ B_2 
    if not np.all(bc == 0):
        raise RuntimeError("Boundary Condition not satisfied! Check incidence matrix again.")

    # lower laplacian 
    L_l = (torch.tensor(B_1).T @ torch.tensor(B_1)).numpy() 

    # upper laplacian 
    L_u = B_2 @ B_2.T 

    # check if the L_l and L_u dimensions are equal to number of edges 
    # first check if they are square 
    if not (L_l.shape[0] == L_l.shape[1] and L_u.shape[0] == L_u.shape[1]):
        raise ValueError("Laplacians are not a square matrix!")
    
    if not (L_l.shape[0] == num_edges and L_u.shape[0] == num_edges): 
        raise ValueError("Size of Laplacians is not equal to the number of edges!")

    return (L_l, L_u), (B_1, B_2)


def get_array_mask(array: np.array, sparsity_prob: float):
    array_cp = copy.deepcopy(array)
    mask = np.random.random(array_cp.shape) >= sparsity_prob 
    return mask 



#####################################################################################

def dl_collate_fn(data_obj_list: List[Data]):
    """
    Custom collate function to handle batching of graphs and compute transformer edge pointers (trafo_ptrs). 
    Args: 
        data_obj_list (list): list of Data objects of pytorch geometric. 

    Returns: 
        databatch: DataBatch object with trafo_ptrs as the added attribute.
    """
    databatch = Batch.from_data_list(data_obj_list)
    num_lines = data_obj_list[0].num_lines # since num_lines are same for all samples 
    num_edges = data_obj_list[0].num_edges

    # trafo_ptrs 
    trafo_ptrs = [num_lines]
    for i in range(1, len(data_obj_list)): 
        trafo_ptrs.append(trafo_ptrs[-1] + num_edges)
    
    databatch.trafo_ptrs = torch.tensor(trafo_ptrs, dtype=int)

    return databatch 


#####################################################################################

def get_device(preferred_device: str = "auto") -> torch.device:

    """Selects the best available device based on user preference and availability.

    Args:
        preferred_device (str): Preferred device ("cpu", "cuda", "mps", or "auto").

    Returns:
        torch.device: The selected device.
    """
    if preferred_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():  # macOS Metal Performance Shaders
            return torch.device("mps")
        else:
            return torch.device("cpu")
        
    elif preferred_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif preferred_device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")  # Default to CPU if the requested device is unavailable

#####################################################################################

def dataset_splitter(dataset:Dataset,
                     batch_size:int,
                     split_list:List[float]=[0.8,0.1,0.1]):
    """Dataset Splitter"""
    # split sizes 
    train_size = int(split_list[0] * len(dataset))
    val_size = int(split_list[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # split dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    plot_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # train_loader = process_trafo_neighbor_loader(trafo_hop, train_loader)
    # val_loader = process_trafo_neighbor_loader(trafo_hop, val_loader)
    # test_loader = process_trafo_neighbor_loader(trafo_hop, test_loader)

    return (train_loader, val_loader, test_loader), plot_loader

def dataset_splitter_fcnn(dataset:Dataset, 
                          batch_size:int, 
                          split_list:List[float]=[0.8, 0.1, 0.1]):
    """Dataset Splitter"""
    # split sizes
    train_size = int(split_list[0] * len(dataset))
    val_size = int(split_list[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # split dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # fcnn dataloaders 
    train_loader = torch_loader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch_loader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch_loader(test_dataset, batch_size=batch_size, shuffle=False)

    # for plotting one graph 
    plot_loader = torch_loader(test_dataset, batch_size=1, shuffle=True)

    return (train_loader, val_loader, test_loader), plot_loader

#####################################################################################

def process_trafo_neighbor_loader(trafo_hop: int, 
                                  loader: DataLoader) -> DataLoader:
    """This function adds the trafo_hop_neighbor attributes based on the batch size. Using the y_trafo_label 
    in the batch and trafo_hop used in the model, this function calculates the trafo_hop neighbors to the trafo_edge and 
    stores it."""
    for batch_list in loader: 
        setattr(batch_list[0], "kaka", torch.tensor(5))
        batch_list[0]._store.kaka = torch.tensor(5)
        node_batch = batch_list[0]

        batchwise_trafo_hop_neighbors = []

        # get the terminal buses of the specific trafo
        trafo_edge = node_batch.y_trafo_label[0][0]

        # since batch can have multiple graphs 
        num_graphs = int(len(node_batch.ptr)-1)

        for i in range(num_graphs):
            # for toy network e.g., ptr = tensor([ 0,  9, 18, 27, 36, 45, 54, 63, 72])
            # get node indices for the current graph
            start, end = node_batch.ptr[i], node_batch.ptr[i+1] # 0, 9 as first graph, based on which trafo_id gets calculated 
            
            # get edges for the current graph 
            mask = (node_batch.edge_index[0] >= start) & (node_batch.edge_index[0] < end)

            # current graph in batch 
            curr_G_edges = node_batch.edge_index[:, mask] - start 

            # build graph 
            curr_G = nx.Graph()
            curr_G_elist = curr_G_edges.T.tolist()
            curr_G.add_edges_from(curr_G_elist)

            # group nodes by shortest path length 
            hop_group_dict_all = {h: [] for h in range(trafo_hop + 1)}
            for node in trafo_edge: # terminal nodes 
                shortest_hops = nx.single_source_shortest_path_length(curr_G, node, cutoff=trafo_hop)
                
                for neighbor, hop in shortest_hops.items():
                    if hop > 0 and neighbor not in trafo_edge:
                        hop_group_dict_all[hop].append(int(neighbor + start))
                    else:
                        hop_group_dict_all[hop].append(int(neighbor + start))
            
            # only consider new and unique nodes for next hop starting with terminal nodes for 0 hop 
            hop_group_dict_uniq = {key: [] for key in hop_group_dict_all.keys()}
            all_uniq_nodes = set() # unique nodes considered for all hops 

            for key in hop_group_dict_all.keys(): 
                hop_uniq_nodes = set(hop_group_dict_all[key]) # no repeatition 
                # if uniq nodes at hop are not already in all_uniq nodes
                if not all_uniq_nodes & hop_uniq_nodes: 
                    all_uniq_nodes.update(hop_group_dict_all[key])
                    hop_group_dict_uniq[key] = hop_group_dict_all[key]
                else: 
                    for and_node in list(all_uniq_nodes & hop_uniq_nodes): 
                        hop_uniq_nodes.remove(and_node)
                    all_uniq_nodes.update(hop_uniq_nodes)
                    hop_group_dict_uniq[key] = list(hop_uniq_nodes)
            batchwise_trafo_hop_neighbors.append(hop_group_dict_uniq) # for all the graphs in batch 
        
        # after calculating it for entire batch, store as a new attribute of batch[0]
        setattr(batch_list[0], 'batchwise_trafo_hop_neighbors', batchwise_trafo_hop_neighbors)

        print(batch_list)
        print("\n \n")
    return loader 





#####################################################################################

def compare_rmse(net: pp.pandapowerNet, 
                 model: nn.Module, 
                 se_iter: int = 5, 
                 v_std: float = 0.001, 
                 p_std: float = 0.2, 
                 is_line_meas: bool = False, 
                 p_line_std: float = 0.2) -> Tuple[Tuple, Tuple]:
    
    """
    Comapare voltage magnitudes pandapower weighted least squares state-estimation and 
    trained model state-estimation using RMSE as the metric. 
    """
    
    # get the stored measurements for consistent power flow results (helps in debugging)
    vm_pfr, va_pfr, p_pfr, q_pfr, p_pfr_line  = use_stored_pfr(net)

    # WLS estimation resulting net (slightly different from rq_1 with stds as parameter)
    net_meas = custom_se(net, 
                         vm_pfr, 
                         p_pfr, 
                         p_pfr_line, 
                         se_iter=se_iter, 
                         v_std=v_std, 
                         p_std=p_std,
                         p_line_std=p_line_std,
                         is_line_meas=is_line_meas)
    
    vm_pu_est = net_meas.res_bus_est.vm_pu.to_numpy()
    va_deg_est = net_meas.res_bus_est.va_degree.to_numpy()

    # WLS RMSE 
    wls_rmse_vm = np.sqrt(np.mean((vm_pfr - vm_pu_est)**2))
    wls_rmse_va = np.sqrt(np.mean((va_pfr - va_deg_est)**2))

    # prepare the pfr (power flow result) for model input
    node_input = torch.tensor(np.column_stack((vm_pfr, 
                                  np.zeros_like(vm_pu_est),
                                  p_pfr,
                                  q_pfr)), dtype=torch.float32) # bcaz torch.tensor(np.array) = float64 not 32.
    
    edge_index_list = get_edge_index_from_ppnet(net)

    model_input = Data(x=node_input,edge_index=edge_index_list)

    with torch.no_grad(): 
        model_out = model(model_input).cpu().numpy()
    
    model_out_vm = model_out[:,0]
    model_out_va = model_out[:,1]
    model_out_p_mw = model_out[:,2]
    model_out_q_mvar = model_out[:,3]

    model_rmse_vm = np.sqrt(np.mean((vm_pfr - model_out_vm)**2))
    model_rmse_va = np.sqrt(np.mean((va_pfr - model_out_va)**2))

    return (wls_rmse_vm, wls_rmse_va), \
        (model_rmse_vm, model_rmse_va), \
            (model_out_vm, model_out_va, model_out_p_mw, model_out_q_mvar), \
                (vm_pfr, va_pfr, p_pfr, q_pfr)

#####################################################################################

def get_rmse(vec1: np.array, 
             vec2: np.array) -> float: 
    
    if isinstance(vec1, torch.Tensor) & isinstance(vec2, torch.Tensor):
        vec1 = vec1.cpu().detach().numpy()
        vec2 = vec2.cpu().detach().numpy()
    else: 
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
    return np.sqrt(np.mean((vec1 - vec2)**2))

#####################################################################################

def get_mae(vec1: np.array, 
            vec2: np.array) -> float: 
    
    if isinstance(vec1, torch.Tensor) & isinstance(vec2, torch.Tensor):
        vec1 = vec1.cpu().detach().numpy()
        vec2 = vec2.cpu().detach().numpy()
    else: 
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

    return np.mean(np.abs(vec1 - vec2))


#####################################################################################

def get_maxae(vec1: np.array, 
            vec2: np.array) -> float: 
    
    if isinstance(vec1, torch.Tensor) & isinstance(vec2, torch.Tensor):
        vec1 = vec1.cpu().detach().numpy()
        vec2 = vec2.cpu().detach().numpy()
    else: 
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

    return np.max(np.abs(vec1 - vec2))


#####################################################################################

def get_nrmse(vec1: np.array, 
            vec2: np.array) -> float: 
    
    if isinstance(vec1, torch.Tensor) & isinstance(vec2, torch.Tensor):
        vec1 = vec1.cpu().detach().numpy()
        vec2 = vec2.cpu().detach().numpy()
    else: 
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

    return get_rmse(vec1, vec2) / np.mean(vec2)


#####################################################################################

def khop_neighborhood_adjacency(net: pp.pandapowerNet, 
                                trafo_idx: int, 
                                k: int = 1) -> np.ndarray:
    
    """
    Returns the k-neighborhood adjacency matrix for the transformer edge in the network.
    
    :param net: pandapower network object
    :param trafo_idx: index of the transformer in the network
    :param k: number of hops to include in the adjacency matrix
    :return masked_adj: A masked adjacency matrix (NumPy array) representing k-neighborhood of the edge.
    :return masked_G: corresponding masked networkx graph
    :return original network networkx graph 
    """

    if trafo_idx in net.trafo.index:
        # extract trafo buses 
        trafo_edge = (net.trafo.hv_bus[trafo_idx], net.trafo.lv_bus[trafo_idx])
    else: 
        raise KeyError(f"Invalid trafo index, total trafos are {len(net.trafo)}")
    
    # create nx graph (pp.topology.create_nx sometimes creates multi-graph, so wrapping with nx.Graph)
    G_unsorted = nx.Graph(pp.topology.create_nxgraph(net, respect_switches=False)) # nodes are not sorted 

    # to keep all edge-weights = 1 (because sometimes the edge-weights are even 0. from pandapower)
    nx.set_edge_attributes(G_unsorted, values=1., name='weight')

    # sorted graph 
    sorted_nodes = sorted(G_unsorted.nodes)
    G = nx.Graph()
    G.add_nodes_from(sorted_nodes)
    G.add_edges_from(G_unsorted.edges)
    
    # k = 0 neighbours (only edge)
    if k == 0: 
        subG = nx.Graph()
        subG.add_edge(trafo_edge[0], trafo_edge[1])
    else: 
        ego_hv = nx.ego_graph(G, trafo_edge[0], radius=k)
        ego_lv = nx.ego_graph(G, trafo_edge[1], radius=k)
        subG = nx.compose(ego_hv, ego_lv)
    
    # now, create a graph adjacency matrix with edge-weights = 1 only for edges in subG else 0 
    maskedG = nx.Graph()
    maskedG.add_nodes_from(G.nodes)

    # iterate through all edges in G 
    for edge in G.edges:
        u,v = edge
        # check if edge in subG
        if not subG.has_edge(u,v):
            maskedG.add_edge(u,v,weight=0)
        else: 
            maskedG.add_edge(u,v,weight=1.)
    
    # get the masked adjacency 
    masked_adj = nx.to_numpy_array(maskedG, nodelist = G.nodes)

    return masked_adj, maskedG, G, trafo_edge


#################################################################################################################

def get_trafo_neighbors(edge_index: torch.Tensor, 
                        trafo_edge: set, 
                        trafo_hop: int, 
                        case_multi: bool = False) -> set:
    if not case_multi: 
        G = nx.Graph()
        edges = edge_index.T.tolist()
        G.add_edges_from(edges)

        # perform BFS on terminal nodes of trafo 
        trafo_neighbors = set()
        for node in trafo_edge: 
            trafo_neighbors.update(nx.single_source_shortest_path_length(G, node, cutoff=trafo_hop))

        return trafo_neighbors
    else: 
        # for multiple transformers
        G = nx.Graph()
        edges = edge_index.T.tolist()
        G.add_edges_from(edges)
        all_trafo_neighbors = {itrafo: set() for itrafo in range(len(trafo_edge))}
        for itrafo, one_edge in enumerate(trafo_edge):
            for node in one_edge: 
                hop_neighbors = nx.single_source_shortest_path_length(G, int(node), cutoff=trafo_hop)
                all_trafo_neighbors[itrafo].update(hop_neighbors)
        return all_trafo_neighbors





#################################################################################################################

def check_dataset(dataset: Dataset):

    return dataset

def check_xee(a,b,c): # all tensors 

    return a, b, c


#################################################################################################################

def tensor_any_nan(*args):
    """Check if any input tensor contains NaN values."""
    nan_indices = [i for i, arg in enumerate(args) if torch.isnan(arg).any().item()]

    return any(torch.isnan(arg).any().item() for arg in args), nan_indices



#################################################################################################################

def load_config(filename: str = 'config.yaml'): 
    path = parent_dir + f"/config/{filename}"
    with open(path, "r") as f: 
        return yaml.safe_load(f)
    


#################################################################################################################
#################################################################################################################
def plot_va_predictions_notebook(plot_loader: DataLoader, trained_model: nn.Module,
                           sampled_input_data: Dict, 
                           fontsize: int = 20):
    """
    Generate bar plots comparing predictions to labels for voltage and angle.
    Displays plots directly in the notebook without saving to a directory.
    
    Args:
        plot_loader: DataLoader containing the graph data
        trained_model: Trained neural network model
        sampled_input_data: Dictionary containing data scaling information
    """
    # Get prediction for a single graph
    single_graph = next(iter(plot_loader))
    with torch.no_grad():
        pred = trained_model(single_graph)
        if trained_model.name == "MultiTapSEGNN":
            pred_se, _ = pred
        else:
            pred_se = pred
            
    print("Plotting Voltage and Angles in PU and Degrees, respectively.\n\n")
    
    # Get scaled/unscaled predictions and labels
    is_scaler = sampled_input_data['scaler_node']
    if is_scaler:
        pred_se_va = inverse_scale(pred_se, scaler=sampled_input_data['scaler_y_label'])
        label_se_va = inverse_scale(single_graph[0].y, scaler=sampled_input_data['scaler_y_label'])
    else:
        pred_se_va = pred_se
        label_se_va = single_graph[0].y
    
    # Determine if we need to plot only first 100 nodes
    plot_first_100_nodes = pred_se_va.shape[0] >= 100
    
    # Get crest colors for consistent color coding
    # cp = "Paired"
    cp = "crest"
    crest_colors = sns.color_palette(cp, 2)
    
    # Create bar plots
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    width = 0.3
    
    if plot_first_100_nodes:
        v_min = torch.min(label_se_va[:100, 0])
        v_max = torch.max(label_se_va[:100, 0])
        xticks = np.arange(100)
        ax[0].bar(xticks - width/2, pred_se_va[:100, 0], width=width, label="Prediction", color=crest_colors[0])
        ax[0].bar(xticks + width/2, label_se_va[:100, 0], width=width, label="Label", color=crest_colors[1])
        ax[0].set_ylim(v_min-0.05, v_max+0.05)
        ax[1].bar(xticks - width/2, pred_se_va[:100, 1], width=width, label="Prediction", color=crest_colors[0])
        ax[1].bar(xticks + width/2, label_se_va[:100, 1], width=width, label="Label", color=crest_colors[1])
    else:
        v_min = torch.min(label_se_va[:, 0])
        v_max = torch.max(label_se_va[:, 0])
        xticks = np.arange(pred_se_va.shape[0])
        ax[0].bar(xticks - width/2, pred_se_va[:, 0], width=width, label="Prediction", color=crest_colors[0])
        ax[0].bar(xticks + width/2, label_se_va[:, 0], width=width, label="Label", color=crest_colors[1])
        ax[0].set_ylim(v_min-0.05, v_max+0.05)
        ax[1].bar(xticks - width/2, pred_se_va[:, 1], width=width, label="Prediction", color=crest_colors[0])
        ax[1].bar(xticks + width/2, label_se_va[:, 1], width=width, label="Label", color=crest_colors[1])
    
    ax[0].set_ylabel("Per Unit", fontsize=fontsize)
    ax[1].set_ylabel("Degree", fontsize=fontsize)
    ax[0].set_xlabel("Buses", fontsize=fontsize)
    ax[0].legend(loc='upper right', bbox_to_anchor=(1, 0.95), borderaxespad=0., fontsize=fontsize)
    ax[1].set_xlabel("Buses", fontsize=fontsize)

    # Update tick font sizes
    ax[0].tick_params(axis='both', labelsize=fontsize)
    ax[1].tick_params(axis='both', labelsize=fontsize)

    ax[1].legend(loc='best', fontsize=fontsize)
    # fig.suptitle(f"SE Predictions vs. Labels")
    fig.tight_layout()
    plt.show()
    
    # Plot joint distribution predictions vs labels
    fig_j, ax_j = plt.subplots(2, 1, figsize=(10, 10))
    
    # Voltage predictions vs labels
    v_pred_label_df = pd.DataFrame(torch.vstack([pred_se_va[:, 0], label_se_va[:, 0]]).T,
                                  columns=['Prediction', 'Label'])
    sns.kdeplot(v_pred_label_df, palette=cp, alpha=0.5, ax=ax_j[0], fill=True)
    ax_j[0].set_xlabel("Voltage Magnitude", fontsize=fontsize)
    ax_j[0].set_ylabel("KDE Density", fontsize=fontsize)
    
    # Angle predictions vs labels
    a_pred_label_df = pd.DataFrame(torch.vstack([pred_se_va[:, 1], label_se_va[:, 1]]).T,
                                  columns=['Prediction', 'Label'])
    sns.kdeplot(a_pred_label_df, palette=cp, alpha=0.5, ax=ax_j[1], fill=True)
    ax_j[1].set_xlabel("Angle Magnitude", fontsize=fontsize)
    ax_j[1].set_ylabel("KDE Density", fontsize=fontsize)

    # Update tick font sizes
    ax_j[0].tick_params(axis='both', labelsize=fontsize)
    ax_j[1].tick_params(axis='both', labelsize=fontsize)
    
    if ax_j[0].get_legend() is not None:
        for text in ax_j[0].get_legend().get_texts():
            text.set_fontsize(fontsize)

    if ax_j[1].get_legend() is not None:
        for text in ax_j[1].get_legend().get_texts():
            text.set_fontsize(fontsize)

    fig_j.tight_layout()
    plt.show()


