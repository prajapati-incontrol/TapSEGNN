import os 
import sys 
from manim import *
import torch.nn as nn 
from torch_geometric.data import Data, Batch, Dataset
import torch

import argparse
import numpy as np
import pandapower as pp
import matplotlib.pyplot as plt 

import time
from datetime import datetime

import torch 
import os 
import sys 
# import logging 
import joblib

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from src.dataset.custom_dataset import NodeEdgeTapDatasetV2
from src.training.trainer import trainer
from utils.model_utils import initialize_model, get_eval_results, save_model
from utils.gen_utils import dataset_splitter, animate_loader, get_rmse, process_trafo_neighbor_loader, get_device, load_config, generate_markdown_report
from utils.ppnet_utils import initialize_network, get_trafo_ids_from_percent
from utils.load_data_utils import load_sampled_input_data, inverse_scale
from utils.plot_utils import plot_two_vec, plot_loss_curves, plot_va_bar


class AnimateTapSEGNN(Scene):
    def construct(self): 
        config = load_config()
        device = get_device(config['device'])
        # saved_models/20250622_142341_MultiTapSEGNN_dft_tnp_BEST

        net = initialize_network(config['data']['net_name'], else_load=config['data']['load_std'], verbose=True)
        

        start_data_load = time.perf_counter()
        sampled_input_data = load_sampled_input_data(sc_type=config['data']['scenario_type'], 
                                                    net=net, 
                                                    num_samples=config['data']['num_samples'],
                                                    noise=config['data']['noise'],
                                                    trafo_ids=config['data']['trafo_ids'],
                                                    scaler=config['data']['scaler'],
                                                    )
        # sampled_input_data = joblib.load(parent_dir + '/sampled_input_data_dft_tnp/0_sc9_DFT_TNP_ALL_TRAFO_EL1e-1_NS_4096.pkl')
        joblib.dump(sampled_input_data, parent_dir + f"/sampled_input_data_dft_tnp/June30_experiment.pkl")
        end_data_load = time.perf_counter() 
        print(f"Dataloading took {end_data_load - start_data_load} seconds.\n\n")

        # instantiate the dataset 
        dataset = NodeEdgeTapDatasetV2(model_name=config['model']['name'], sampled_input_data=sampled_input_data)

        all_loaders, plot_loader = dataset_splitter(dataset,
                                    batch_size=config['loader']['batch_size'], 
                                    split_list=config['loader']['split_list'])
        
        anim_loader = animate_loader(dataset)        

        model = initialize_model(model_name=config['model']['name'],
                dataset=dataset,
                node_out_features=config['model']['node_out_features'],
                list_node_hidden_features=config['model']['list_node_hidden_features'],
                k_hop_node=config['model']['k_hop_node'],
                edge_out_features=config['model']['edge_out_features'], 
                list_edge_hidden_features=config['model']['list_edge_hidden_features'],
                k_hop_edge=config['model']['k_hop_edge'],
                trafo_hop=config['model']['trafo_hop'],
                edge_index_list=sampled_input_data['edge_index'],
                gat_out_features=config['model']['gat_out_features'],
                gat_head=config['model']['gat_head'],
                bias=config['model']['bias'], 
                normalize=config['model']['normalize'], 
                device=device,
                ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Total number of parameters of model {model}: {total_params}')

        optimizer = torch.optim.Adam(model.parameters(),lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
        
        schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, 
                                                            mode='min',
                                                            factor=0.1, 
                                                            # patience=1,
                                                            min_lr=config['training']['schedular_min_lr'])

        all_losses = trainer(model=model, 
                            train_loader=all_loaders[0], 
                            val_loader=all_loaders[1], 
                            test_loader=all_loaders[2], 
                            optimizer=optimizer,
                            schedular=schedular,
                            num_epoch=config['training']['num_epochs'],
                            early_stopping=config['training']['early_stopping'],
                            val_patience=config['training']['val_patience'], 
                            tap_weight=config['training']['loss_tap_weight'], 
                            device=device)

        # checkpoint = torch.load(parent_dir + '/saved_models/20250602_102933_MultiTapSEGNN_ltw0.1_trafo100.pth')

        # model.load_state_dict(checkpoint)
        save_model(model, path=parent_dir + f"/saved_models/June30_experiment.pth")
        
        node_label_features = inverse_scale(sampled_input_data['y_label'], 
                                            scaler=sampled_input_data['scaler_y_label'])
        
        node_pred_features = torch.zeros_like(node_label_features)

        # pred_tap_logits = dict()

        with torch.no_grad(): 
            for graph_id, graph in enumerate(anim_loader): 
                node_pred_features[graph_id], _ = model(graph)
        
        node_pred_features = inverse_scale(node_pred_features, 
                                           scaler=sampled_input_data['scaler_y_label'])
        
        # input 
        feature_id = 1

        data_list = []
        data_list = [
            (
                node_label_features[perm, :, feature_id].tolist(), 
                node_pred_features[perm, :, feature_id].tolist(), 
                f"Sample {perm}"
            )
            for perm in range(len(dataset))
        ]

        lenx = len(net.bus.index)

        # Set the aspect ratio
        self.camera.frame_width = 42  # Set the width of the camera frame (x-axis size)
        self.camera.frame_height = 20  # Set the height of the camera frame (y-axis size)

        plane = NumberPlane(
                x_range = (0, lenx, 5),
                # y_range = (y_min, y_max, float(y_max - y_min)*0.25),
                y_range = (-2.0, 2.0, 0.8),
                # y_range = (1.0, 1.12, 0.05),
                x_length = 34,
                y_length = 12,
                axis_config={"include_numbers": True,"font_size": 72},
            ).to_edge(DOWN)

        plane.center()

        # labels 
        xlabel = Text("Buses", font_size=72).move_to(DOWN*8)
        match feature_id: 
            case 0: 
                ylabel = Text("Voltage Mag. (pu)", font_size=72).rotate(90*DEGREES).move_to(LEFT*18)
            case 1: 
                ylabel = Text("Voltage Angle (Deg)", font_size=72).rotate(90*DEGREES).move_to(LEFT*18)
            
        # legends 
        d_mod1 = Dot(radius=0.24).move_to(UP*8.5 + RIGHT*16)
        d_mod2 = Dot(radius=0.24).move_to(UP*8.5 + RIGHT*14)
        # model line 
        l_model = Line(d_mod1.get_center(), d_mod2.get_center(), color=GOLD_E).set_stroke(width = 7)

        d_wls1 = Dot(radius=0.24).move_to(UP*7 + RIGHT*16)
        d_wls2 = Dot(radius=0.24).move_to(UP*7 + RIGHT*14)
        # model line 
        l_wls = Line(d_wls1.get_center(), d_wls2.get_center(), color=BLUE).set_stroke(width = 7)
        

        title = Text("TapSEGNN on Sample", font_size=72).move_to(UP*8 + LEFT)

        leg1 = Text("Predictions", font_size=72).move_to(UP*8.5 + RIGHT*10)
        leg2 = Text("True Values", font_size=72).move_to(UP*7 + RIGHT*10)
        self.play(Write(title), Create(plane), Write(xlabel), Write(ylabel))
        self.wait(2)
        self.play(Write(leg1), Write(leg2))
        self.play(Create(l_model), Create(l_wls))



        for i, (input_data, output_data, title) in enumerate(data_list[::40]):
                          
            line_graph_input = plane.plot_line_graph(
                x_values = [i for i in range(lenx)],
                y_values = input_data,
                stroke_width = 12,
            ).set_stroke(color=BLUE, width=7)

            line_graph_output = plane.plot_line_graph(
                x_values = [i for i in range(lenx)],
                y_values = output_data,
                line_color=GOLD_E,
                stroke_width = 12,
            ).set_stroke(color=GOLD_E, width=7)

            sample_id = Text(str(i+1), font_size=72).move_to(UP*8.1+4.5*RIGHT)
            self.add(line_graph_input, line_graph_output, sample_id)
            self.wait(0.20)
            self.remove(line_graph_input, line_graph_output, sample_id)

if __name__ == "__main__":
    scene = AnimateTapSEGNN()
        
        
        

