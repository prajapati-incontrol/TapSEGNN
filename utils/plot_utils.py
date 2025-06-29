import networkx as nx 
import numpy as np
import pandapower as pp
import warnings 
import torch
import copy
import os
import pickle
import pandas as pd 
from typing import Tuple, List, Dict
from torch_geometric.data import Data, Batch, Dataset
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import pylab
import time 
import sys
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch
import seaborn as sns 

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from utils.load_data_utils import inverse_scale, load_sampled_input_data
from utils.ppnet_utils import initialize_network

def plot_param_distribution(net_name: str,
                            filename: str,
                            scaler: bool = False, 
                            get_rxbg_pu: bool = False, 
                            ): 
    """
    Plot the parameter distribution for given network.
    """
    net = initialize_network(net_name)

    num_lines = len(net.line.index)
    num_trafos = len(net.trafo.index)

    num_samples = 1 
    noise_bool = False 
    sc_type = 1 
    
    sampled_input_data = load_sampled_input_data(sc_type=sc_type, 
                                            net=net, 
                                            num_samples=num_samples,
                                            p_std=0.0,
                                            noise=noise_bool,
                                            scaler=scaler,
                                            )
    
    r_pu = sampled_input_data['edge_input_feat'][:,:,0].numpy()
    x_pu = sampled_input_data['edge_input_feat'][:,:,1].numpy()
      
    b_pu = sampled_input_data['edge_input_feat'][:,:,2].numpy()
    g_pu = sampled_input_data['edge_input_feat'][:,:,3].numpy()

    # Create a list of colors: first 5 red, rest blue
    bar_colors = ['tab:red'] * num_lines + ['tab:blue'] * num_trafos

    fig, ax = plt.subplots(2,2,figsize=(10,10))
    ax = ax.flatten()
    xticks = np.arange(r_pu.shape[1])
    ax[0].bar(xticks, r_pu.squeeze(), color=bar_colors)
    ax[0].set_ylabel("R [pu]")
    ax[0].set_xlabel("Branch Index")

    ax[1].bar(xticks, x_pu.squeeze(), color=bar_colors)
    ax[1].set_ylabel("X [pu]")
    ax[1].set_xlabel("Branch Index")
    
    ax[2].bar(xticks, b_pu.squeeze(), color=bar_colors)
    ax[2].set_ylabel("B [pu]")
    ax[2].set_xlabel("Branch Index")

    ax[3].bar(xticks, g_pu.squeeze(), color=bar_colors)
    ax[3].set_ylabel("G [pu]")
    ax[3].set_xlabel("Branch Index")

    # create super legend 
    legend_elements = [
        Patch(facecolor='red', label='lines'), 
        Patch(facecolor='blue',label='trafos')
    ]

    fig.legend(handles=legend_elements)

    fig.suptitle(f"Branch parameters for {net_name}")

    net_dir = net_name 
    os.makedirs(parent_dir + f"/notebooks/rq8_results/{net_dir}",exist_ok=True)
    fig.savefig(parent_dir + f"/notebooks/rq8_results/{net_dir}/{filename}.pdf", bbox_inches='tight')
    # ani.save(parent_dir + f"/notebooks/rq4_results/{net_dir}/{net_name} VAPQ_scale{scaler}.mp4", writer="ffmpeg", fps=5)
    plt.show()
    plt.close(fig)

    if get_rxbg_pu == True: 
        return r_pu, x_pu, b_pu, g_pu






#####################################################################################

def animate_vapq_net(net_name: str, 
                     num_samples: int, 
                     load_std: float = 0.0, 
                     noise: float = 0.0, 
                     scaler: bool = False): 
    """
    Animates the Voltage, Angle, Active Power and Reactive Power Magnitude 
    over all buses acros all samples as frames of animation. 
    """

    net = initialize_network(net_name)

    sampled_input_data = load_sampled_input_data(sc_type=1, 
                                                 net=net, 
                                                 num_samples=num_samples, 
                                                 p_std=load_std, 
                                                 noise=noise, 
                                                 scaler=scaler)
    
    vm_pu_2d = sampled_input_data['node_input_feat'][:, :, 0]
    vm_min, vm_max = vm_pu_2d.min().item(), vm_pu_2d.max().item()
    a_deg = sampled_input_data["node_input_feat"][:,:,1]
    a_deg_min, a_deg_max = a_deg.min().item(), a_deg.max().item()
    p_mw = sampled_input_data['node_input_feat'][:,:,2]
    p_mw_min, p_mw_max = p_mw.min().item(), p_mw.max().item()
    q_mvar = sampled_input_data['node_input_feat'][:,:,3]
    q_mvar_min, q_mvar_max = q_mvar.min().item(), q_mvar.max().item()
    
    num_buses = sampled_input_data['node_input_feat'].shape[1]

    fig, ax = plt.subplots(2,2,figsize=(10,10))
    ax = ax.flatten()

    def update(idx): 
        width = 0.3 

        ax[0].clear() 
        xticks = np.arange(num_buses)
        ax[0].bar(xticks, vm_pu_2d[idx, :], width=width, label="|V|")
        ax[0].set_ylabel("|V| [p.u.]")
        ax[0].set_xlabel("Buses")
        if scaler: 
            ax[0].set_ylim(vm_min*1.1, vm_max*1.1)
        else: 
            ax[0].set_ylim(0.0, vm_max*1.1)
        ax[0].legend()

        ax[1].clear()
        ax[1].bar(xticks, a_deg[idx,:], width=width, label=r"$\theta$")
        ax[1].set_ylabel(r"$\theta$ [deg]")
        ax[1].set_xlabel("Buses")
        ax[1].set_ylim(a_deg_min*1.1, a_deg_max*1.1)
        ax[1].legend()

        ax[2].clear()
        ax[2].bar(xticks, p_mw[idx,:], width=width, label="P")
        ax[2].set_ylabel("P [MW]")
        ax[2].set_xlabel("Buses")
        ax[2].set_ylim(p_mw_min*1.1, p_mw_max*1.1)
        ax[2].legend()

        ax[3].clear()
        ax[3].bar(xticks, q_mvar[idx,:], width=width, label="Q")
        ax[3].set_ylabel("Q [MVAR]")
        ax[3].set_xlabel("Buses")
        ax[3].set_ylim(q_mvar_min*1.1, q_mvar_max*1.1)
        ax[3].legend()

        fig.suptitle(f"Sample {idx}")
    
    ani = FuncAnimation(fig, update, frames=num_samples, interval=100, repeat=True)

    net_dir = net_name 
    os.makedirs(parent_dir + f"/notebooks/rq4_results/{net_dir}",exist_ok=True)
    ani.save(parent_dir + f"/notebooks/rq4_results/{net_dir}/{net_name} VAPQ_scale{scaler}.mp4", writer="ffmpeg", fps=5)
    plt.close(fig)

#####################################################################################


def animate_distribution_2d(data_2d: np.ndarray | torch.Tensor, 
                         axis: int, 
                         title: str,
                         log_dir_path: str, 
                         fps: int = 10,
    ):
    """
    Animates the distribution of the array across given axis.
    """
    assert axis == 0 | axis == 1, "Invalid axis argument."
    
    item_min = data_2d.min().item()
    item_max = data_2d.max().item()

    fig, ax = plt.subplots(figsize=(10,10))

    def update(idx):
        ax.clear()
        if axis == 0: # across buses/columns
            idx_data = data_2d[idx, :].numpy()
            title_str = f"{title} at sample {idx} for all {data_2d.shape[1]} buses"
        else: # across all samples for a given bus
            idx_data = data_2d[:, idx].numpy()
            title_str = f"{title} at bus {idx} for all {data_2d.shape[0]} samples."
        sns_violin = sns.violinplot(y=idx_data, ax=ax, color='skyblue',inner='box')
        # set opacity 
        for art in ax.collections: 
            art.set_alpha(0.5)
        ax.set_title(title_str)
        ax.set_ylim(item_min, item_max)
    
    # Create animation: 9 frames (one per column)
    ani = FuncAnimation(fig, update, frames=data_2d.shape[axis], interval=100, repeat=True)

    os.makedirs(log_dir_path, exist_ok=True)

    ani.save(log_dir_path + f"/{title}.mp4", writer="ffmpeg", fps=fps)
    plt.close(fig) 


#####################################################################################

def plot_va_bar(model: nn.Module, 
                test_loader: DataLoader,
                net_name: str, 
                sampled_input_data: dict, 
                va_scaled: bool = False):
    """Plots the voltage and angle of the model for a single graph."""
    batch = next(iter(test_loader))
    with torch.no_grad():
        if "Tap" in model.name: 
            pred_se, _ = model(batch)
        else: 
            pred_se = model(batch) 

    if va_scaled: # scaled v and a 
        pred_v = pred_se[:,0].numpy()
        pred_a = pred_se[:,1].numpy()
        label_v = batch[0].y[:,0].numpy()
        label_a = batch[0].y[:,1].numpy()
    else: 
        # predicted voltage inverse-scaled
        pred_se_unscaled = inverse_scale(pred_se, scaler=sampled_input_data['scaler_y_label'])

        # labels inverse-scaled 
        y_unscaled = inverse_scale(batch[0].y, scaler=sampled_input_data['scaler_y_label'])

        pred_v = pred_se_unscaled[:,0].numpy()
        pred_a = pred_se_unscaled[:,1].numpy()
        label_v = y_unscaled[:,0].numpy()
        label_a = y_unscaled[:,1].numpy()
    
    first_50 = pred_v.shape[0] >= 100

    if first_50: 
        fig, ax = plt.subplots(2,1,figsize=(10,10))
        width = 0.3 
        xticks = np.arange(len(pred_v[:50]))
        ax[0].bar(xticks-width/2, pred_v[:50], width=width, label="Pred V")
        ax[0].bar(xticks+width/2, label_v[:50], width=width, label="Label V")
        ax[0].set_ylabel("Magnitude")
        ax[0].set_xlabel("Buses")
        ax[0].legend()
        
        ax[1].bar(xticks-width/2, pred_a[:50], width=width, label="Pred A")
        ax[1].bar(xticks+width/2, label_a[:50], width=width, label="Label A")
        ax[1].set_ylabel("Magnitude")
        ax[1].set_xlabel("Buses")
        ax[1].legend()
        fig.suptitle(f"{net_name} SE Predictions vs. Labels")
        
        fig.tight_layout()
        plt.show()
    else: 
        fig, ax = plt.subplots(2,1,figsize=(10,10))
        width = 0.3 
        xticks = np.arange(len(pred_v))
        ax[0].bar(xticks-width/2, pred_v, width=width, label="Pred V")
        ax[0].bar(xticks+width/2, label_v, width=width, label="Label V")
        ax[0].set_ylabel("Magnitude")
        ax[0].set_xlabel("Buses")
        ax[0].legend()
        
        ax[1].bar(xticks-width/2, pred_a, width=width, label="Pred A")
        ax[1].bar(xticks+width/2, label_a, width=width, label="Label A")
        ax[1].set_ylabel("Magnitude")
        ax[1].set_xlabel("Buses")
        ax[1].legend()
        fig.suptitle(f"{net_name} SE Predictions vs. Labels")
        
        fig.tight_layout()
        plt.show()
        


#####################################################################################

def plot_loss_curves(train_losses: List,
                    val_losses:List,
                    model_name:nn.Module,
                    node_hidden_list:List,
                    k_hop:int,
                    edge_hidden_list:List=None,
                    k_hop_edge:int=None,
                    plot_logarithmic: bool = False,
                    last_epochs: int = None): 
    """
    Plots training and validation loss curves. If their averages of last 5 epochs are the same, plot 
    in a single figure, else seperate subplots. 
    #TODO Use median if loss curve have extreme spikes 
    #TODO Use final loss value if interested in end performance 
    """

    # mean of last five epochs
    train_stat = np.mean(train_losses[-5:])
    val_stat = np.mean(val_losses[-5:])
    save_path = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(save_path): 
        os.makedirs(save_path)
    
    file_prefix = f"Loss_"

    # if means are close by 1.0 
    # if np.isclose(train_stat, val_stat, rtol=0.4): 
    if not plot_logarithmic:
        if np.abs(train_stat - val_stat) <= 1:
            if last_epochs: 
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(train_losses[-last_epochs:], label="Training Loss")
                ax.plot(val_losses[-last_epochs:], label="Validation Loss")
                ax.set_xlabel("Epochs")
                ax.set_ylabel("Loss")
                ax.legend()
                fig.suptitle(f"Model {model_name} - Hidden Layers {node_hidden_list} with k-hop {k_hop}")
                fig.tight_layout()
                save_file = os.path.join(save_path, f"{file_prefix}.png")
            else: 
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(train_losses, label="Training Loss")
                ax.plot(val_losses, label="Validation Loss")
                ax.set_xlabel("Epochs")
                ax.set_ylabel("Loss")
                ax.legend()
                fig.suptitle(f"Model {model_name} - Hidden Layers {node_hidden_list} with k-hop {k_hop}")
                fig.tight_layout()
                save_file = os.path.join(save_path, f"{file_prefix}.png")
        else: 
            if last_epochs: 
                # seperate plots 
                fig, ax = plt.subplots(2, 1, figsize=(10, 6))
                ax[0].plot(train_losses[-last_epochs:])
                ax[0].set_xlabel("Epochs")
                ax[0].set_ylabel("Training Loss")
                ax[0].set_title("Training Loss Curve")

                ax[1].plot(val_losses[-last_epochs:])
                ax[1].set_xlabel("Epochs")
                ax[1].set_ylabel("Validation Loss")
                ax[1].set_title("Validation Loss Curve")

                fig.suptitle(f"Model {model_name} - Hidden Layers {node_hidden_list} with k-hop {k_hop}")
                fig.tight_layout()
                save_file = os.path.join(save_path, f"{file_prefix}_separate.png")
            else: 
                # seperate plots 
                fig, ax = plt.subplots(2, 1, figsize=(10, 6))
                ax[0].plot(train_losses)
                ax[0].set_xlabel("Epochs")
                ax[0].set_ylabel("Training Loss")
                ax[0].set_title("Training Loss Curve")

                ax[1].plot(val_losses)
                ax[1].set_xlabel("Epochs")
                ax[1].set_ylabel("Validation Loss")
                ax[1].set_title("Validation Loss Curve")

                fig.suptitle(f"Model {model_name} - Hidden Layers {node_hidden_list} with k-hop {k_hop}")
                fig.tight_layout()
                save_file = os.path.join(save_path, f"{file_prefix}_separate.png")
        plt.savefig(save_file, bbox_inches='tight')
        plt.show()
    else: 
        if np.abs(train_stat - val_stat) <= 1:
            if last_epochs: 
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(train_losses[-last_epochs:], label="Training Loss")
                ax.plot(val_losses[-last_epochs:], label="Validation Loss")
                ax.set_xlabel("Epochs")
                ax.set_ylabel("Loss")
                ax.set_yscale('log')
                ax.legend()
                fig.suptitle(f"Model {model_name} - Hidden Layers {node_hidden_list} with k-hop {k_hop} (logarithmic)")
                fig.tight_layout()
                save_file = os.path.join(save_path, f"{file_prefix}_log.png")
            else: 
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(train_losses, label="Training Loss")
                ax.plot(val_losses, label="Validation Loss")
                ax.set_xlabel("Epochs")
                ax.set_ylabel("Loss")
                ax.set_yscale('log')
                ax.legend()
                fig.suptitle(f"Model {model_name} - Hidden Layers {node_hidden_list} with k-hop {k_hop} (logarithmic)")
                fig.tight_layout()
                save_file = os.path.join(save_path, f"{file_prefix}_log.png")
        else: 
            if last_epochs: 
                # seperate plots 
                fig, ax = plt.subplots(2, 1, figsize=(10, 6))
                ax[0].plot(train_losses[-last_epochs:])
                ax[0].set_yscale('log')
                ax[0].set_xlabel("Epochs")
                ax[0].set_ylabel("Training Loss")
                ax[0].set_title("Training Loss Curve")

                ax[1].plot(val_losses[-last_epochs:])
                ax[1].set_yscale('log')
                ax[1].set_xlabel("Epochs")
                ax[1].set_ylabel("Validation Loss")
                ax[1].set_title("Validation Loss Curve")

                fig.suptitle(f"Model {model_name} - Hidden Layers {node_hidden_list} with k-hop {k_hop} (logarithmic)")
                fig.tight_layout()
                save_file = os.path.join(save_path, f"{file_prefix}_separate_log.png")
            else: 
                # seperate plots 
                fig, ax = plt.subplots(2, 1, figsize=(10, 6))
                ax[0].plot(train_losses)
                ax[0].set_yscale('log')
                ax[0].set_xlabel("Epochs")
                ax[0].set_ylabel("Training Loss")
                ax[0].set_title("Training Loss Curve")

                ax[1].plot(val_losses)
                ax[1].set_yscale('log')
                ax[1].set_xlabel("Epochs")
                ax[1].set_ylabel("Validation Loss")
                ax[1].set_title("Validation Loss Curve")

                fig.suptitle(f"Model {model_name} - Hidden Layers {node_hidden_list} with k-hop {k_hop} (logarithmic)")
                fig.tight_layout()
                save_file = os.path.join(save_path, f"{file_prefix}_separate_log.png")
        plt.savefig(save_file, bbox_inches='tight')
        pylab.show()


#####################################################################################

def plot_two_vec(vec1: np.array, 
                vec2: np.array,
                vec1_name: str, 
                vec2_name: str):
    """Plots two vectors as bar graph"""
    width = 0.3 
    fig, ax = plt.subplots(figsize=(10,10))

    xticks = np.arange(len(vec1))
    ax.bar(xticks-width/2, vec1, width=width, label=vec1_name)
    ax.bar(xticks+width/2, vec2, width=width, label=vec2_name)
    ax.set_ylabel("Magnitude")
    ax.set_xlabel("Vectors")
    ax.legend()
    fig.tight_layout()
    plt.show()