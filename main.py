import os 
import sys 
import torch
import time
from datetime import datetime
import random 
import numpy as np 

parent_dir = os.getcwd()

from src.dataset.custom_dataset import NodeEdgeTapDatasetV2, GenDataset, DiscDataset
from src.model.graph_model import NEGATGenerator, DiffPoolDiscriminator
from training.trainer import trainer, train_GAN
from utils.model_utils import initialize_model, get_eval_results, impute_dataset_with_model_G
from utils.gen_utils import dataset_splitter, get_device, load_config, generate_markdown_report_and_save_model, generate_markdown_report_GAN_and_save_model
from utils.ppnet_utils import initialize_network
from utils.load_data_utils import load_sampled_input_data


def main(): 
    start_time_main = time.perf_counter()
    print("\n=========================== Starting Main ===========================\n")
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    yaml_config = load_config('config_gan.yaml')
    config_tapsegnn = load_config('config.yaml')
    device = get_device(yaml_config['device'])

    print(f"\n=========================== Using device: {device} ===========================\n")

    print("\n=========================== Loading the network ===========================\n")
    net = initialize_network(net_name=yaml_config['data']['net_name'],
                         load_std=yaml_config['data']['load_std'])
    
    print("\n=========================== Initializing GANs ===========================\n")
    # data for generator model
    start_data_load = time.perf_counter()
    sampled_input_data_G = load_sampled_input_data(sc_type=yaml_config['data']['gen_scenario_type'], 
                                                   net=net, 
                                                   num_samples=yaml_config['data']['num_samples'], 
                                                   noise=yaml_config['data']['noise'])
    end_data_load = time.perf_counter() 
    print(f"\n=========================== Dataloading took {end_data_load - start_data_load} seconds ===========================\n")
    # data for discriminator model 
    sampled_input_data_D = load_sampled_input_data(sc_type=yaml_config['data']['dis_scenario_type'], 
                                                   net=net, 
                                                   num_samples=yaml_config['data']['num_samples'], 
                                                   noise=yaml_config['data']['noise'])
    
    print("\n =========================== Loading dataset for Generator Model ===========================\n")
    dataset_G = GenDataset(model_name=yaml_config['model_G']['name'], 
                       sampled_input_data=sampled_input_data_G)

    (train_loader_G, val_loader_G, test_loader_G), _ = dataset_splitter(dataset_G, 
                                                                        batch_size=yaml_config['loader']['batch_size'])
    
    print("\n=========================== Loading dataset for Discriminator Model ===========================\n")
    dataset_D = DiscDataset(sampled_input_data=sampled_input_data_D)

    (train_loader_D, val_loader_D, test_loader_D), _ = dataset_splitter(dataset_D,
                                                                        batch_size=yaml_config['loader']['batch_size'])

    print("\n=========================== Training GAN ===========================\n")
    # instantiate model, optimizer and schedular for Generator 
    model_G = NEGATGenerator(node_input_features=dataset_G[0][0].x.shape[-1], 
                        list_node_hidden_features=yaml_config['model_G']['list_node_hidden_features'],
                        node_out_features=yaml_config['model_G']['node_out_features'],  
                        k_hop_node=yaml_config['model_G']['k_hop_node'], 
                        edge_input_features=dataset_G[0][1].x.shape[-1], 
                        list_edge_hidden_features=yaml_config['model_G']['list_edge_hidden_features'],  
                        edge_output_features=yaml_config['model_G']['edge_out_features'],  
                        k_hop_edge=yaml_config['model_G']['k_hop_edge'],
                        gat_out_features=yaml_config['model_G']['gat_out_features'],  
                        gat_head=yaml_config['model_G']['gat_head'], 
                        device=device)

    optimizer_G = torch.optim.Adam(model_G.parameters(), 
                                lr=yaml_config['training_G']['lr'], 
                                weight_decay=yaml_config['training_G']['weight_decay'])

    schedular_G = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_G, 
                                                        mode='min', 
                                                        factor=0.1, 
                                                        min_lr=yaml_config['training_G']['schedular_min_lr'])

    total_params_G = sum(p.numel() for p in model_G.parameters() if p.requires_grad)
    print(f'Total number of parameters of model {model_G}: {total_params_G}')

    # instantiate model, optimizer and schedular for Discriminator 
    model_D = DiffPoolDiscriminator(in_channel=dataset_D[0].x.shape[-1], 
                                hidden_channel=yaml_config['model_D']['hidden_channel'], 
                                out_channel=yaml_config['model_D']['out_channel'], 
                                num_nodes=len(net.bus.index))

    total_params_D = sum(p.numel() for p in model_D.parameters() if p.requires_grad)
    print(f'Total number of parameters of model {model_D}: {total_params_D}')

    optimizer_D = torch.optim.Adam(model_D.parameters(), 
                                lr=yaml_config['training_D']['lr'], 
                                weight_decay=yaml_config['training_D']['weight_decay'])

    schedular_D = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_D, 
                                                        mode='min', 
                                                        factor=0.1, 
                                                        min_lr=yaml_config['training_D']['schedular_min_lr'])

    all_losses = train_GAN(model_G=model_G, 
                        model_D=model_D, 
                        all_loader_G= [train_loader_G, val_loader_G, test_loader_G], 
                        all_loader_D= [train_loader_D, val_loader_D, test_loader_D],  
                        optimizer_G=optimizer_G, 
                        optimizer_D=optimizer_D, 
                        schedular_G=schedular_G, 
                        schedular_D=schedular_D, 
                        num_epoch=yaml_config['training_GAN']['num_epoch'], 
                        disc_iter=yaml_config['training_GAN']['disc_iter'], 
                        gen_iter=yaml_config['training_GAN']['gen_iter'],
                        feature_matching=yaml_config['training_GAN']['feature_matching'], 
                        label_smoothing=yaml_config['training_GAN']['label_smoothing'],
                        device=device)  

    print("\n=========================== GAN training completed successfully! Generating a report ===========================\n")
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir_gan = f"{parent_dir}/results/{current_time}/GAN"
    generated_data, simulated_v_pf = generate_markdown_report_GAN_and_save_model(report_dir=report_dir_gan,
                                                                yaml_config=yaml_config, 
                                                                train_g_losses=all_losses['train_g_losses'], 
                                                                train_d_losses=all_losses['train_d_losses'], 
                                                                train_d_accuracies=all_losses['train_d_accuracies'],  
                                                                test_loader_G=test_loader_G, 
                                                                model_G=model_G, 
                                                                sampled_input_data_G=sampled_input_data_G, 
                                                                return_data=True,
                                                                save_model_bool=True,
                                                            )
    
    print("\n=========================== Report saved. ===========================\n")

    print("\n=========================== Using the trained Generator Model to create samples for TapSEGNN... ===========================\n")
    dataset_w_mm = NodeEdgeTapDatasetV2(model_name=config_tapsegnn['model']['name'], 
                               sampled_input_data=sampled_input_data_G, 
                               missing_measurements=True)
    
    # dataset with imputed pv and p edge
    sampled_input_data_imputed = impute_dataset_with_model_G(model_G=model_G, 
                                                            dataset_w_mm=dataset_w_mm, 
                                                            sampled_input_data=sampled_input_data_G)

    dataset = NodeEdgeTapDatasetV2(model_name=config_tapsegnn['model']['name'], 
                                sampled_input_data=sampled_input_data_imputed)
    
    print("=========================== Training the TapSEGNN model... ===========================\n")
    all_loaders, plot_loader = dataset_splitter(dataset,
                                    batch_size=config_tapsegnn['loader']['batch_size'], 
                                    split_list=config_tapsegnn['loader']['split_list'])

    model = initialize_model(model_name=config_tapsegnn['model']['name'],
                        dataset=dataset,
                        node_out_features=config_tapsegnn['model']['node_out_features'],
                        list_node_hidden_features=config_tapsegnn['model']['list_node_hidden_features'],
                        k_hop_node=config_tapsegnn['model']['k_hop_node'],
                        edge_out_features=config_tapsegnn['model']['edge_out_features'], 
                        list_edge_hidden_features=config_tapsegnn['model']['list_edge_hidden_features'],
                        k_hop_edge=config_tapsegnn['model']['k_hop_edge'],
                        trafo_hop=config_tapsegnn['model']['trafo_hop'],
                        edge_index_list=sampled_input_data_G['edge_index'],
                        gat_out_features=config_tapsegnn['model']['gat_out_features'],
                        gat_head=config_tapsegnn['model']['gat_head'],
                        bias=config_tapsegnn['model']['bias'], 
                        normalize=config_tapsegnn['model']['normalize'], 
                        device=device,
                        ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of parameters of model {model}: {total_params}')

    optimizer = torch.optim.Adam(model.parameters(),
                                lr=config_tapsegnn['training']['lr'], 
                                weight_decay=config_tapsegnn['training']['weight_decay'])
            
    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, 
                                                        mode='min',
                                                        factor=0.1, 
                                                        patience=1,
                                                        min_lr=config_tapsegnn['training']['schedular_min_lr'])

    all_losses = trainer(model=model, 
                        train_loader=all_loaders[0], 
                        val_loader=all_loaders[1], 
                        test_loader=all_loaders[2], 
                        optimizer=optimizer,
                        schedular=schedular,
                        num_epoch=config_tapsegnn['training']['num_epochs'],
                        early_stopping=config_tapsegnn['training']['early_stopping'],
                        val_patience=config_tapsegnn['training']['val_patience'], 
                        tap_weight=config_tapsegnn['training']['loss_tap_weight'], 
                        device=device)

    results = get_eval_results(test_loader=all_loaders[2],
                                    tap_weight=config_tapsegnn['training']['loss_tap_weight'], 
                                trained_model=model, 
                                scaler=sampled_input_data_G['scaler_y_label'])

    generate_markdown_report_and_save_model(current_time=current_time,
                            all_losses=all_losses, 
                            config=config_tapsegnn, 
                            results=results, 
                            plot_loader=plot_loader, 
                            trained_model=model, 
                            sampled_input_data=sampled_input_data_G, 
                            usetex=True, 
                            save_model_bool=True)

    end_time_main = time.perf_counter()
    print(f"\n=========================== Main run complete! Took {end_time_main - start_time_main} seconds. ===========================\n")


if __name__ == "__main__":
    main()
