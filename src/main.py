import time
from datetime import datetime
import torch 
import os 
import sys 

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from dataset.custom_dataset import NodeEdgeTapDatasetV2
from training.trainer import trainer
from utils.model_utils import initialize_model, get_eval_results
from utils.gen_utils import dataset_splitter, get_device, load_config, generate_markdown_report_and_save_model
from utils.ppnet_utils import initialize_network
from utils.load_data_utils import load_sampled_input_data

torch.manual_seed(0)

def main():

    config = load_config()
    device = get_device(config['device']) 

    print(f"Using device: {device}")

    net = initialize_network(config['data']['net_name'], load_std=config['data']['load_std'], verbose=True)

    start_data_load = time.perf_counter()
    sampled_input_data = load_sampled_input_data(sc_type=config['data']['scenario_type'], 
                                                net=net, 
                                                num_samples=config['data']['num_samples'],
                                                noise=config['data']['noise'],
                                                trafo_ids=config['data']['trafo_ids'],
                                                scaler=config['data']['scaler'],
                                                )
    
    # joblib.dump(sampled_input_data, os.getcwd() + f"/sampled_input_data_mvo/0_sc9_DFT_TNP_ALL_TRAFO_EL0.1._NS_4096pkl")
    # break 
    # sampled_input_data = joblib.load(os.getcwd() + f"/sampled_input_data_dft_tnp/sc9_ns32768_el_0.1.pkl")
    end_data_load = time.perf_counter() 
    print(f"Dataloading took {end_data_load - start_data_load} seconds.\n\n")
    # print(sampled_input_data['y_trafo_label'])
    # exit()
    time.sleep(3)
    
    # instantiate the dataset 
    dataset = NodeEdgeTapDatasetV2(model_name=config['model']['name'], sampled_input_data=sampled_input_data)
    # print(dataset[0])
    # exit()

    all_loaders, plot_loader = dataset_splitter(dataset,
                                    batch_size=config['loader']['batch_size'], 
                                    split_list=config['loader']['split_list'])
    # print(len(test_loader))
    # print(len(train_loader))
    # print(len(val_loader))
    # print(len(dataset))
    # exit()
    
    # all_trafo_hops = [5]
    # for trafo_hop in all_trafo_hops: 
    #     config['model']['trafo_hop'] = trafo_hop

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
    # time.sleep(3)
    
    # for name, param in model.named_parameters():
    #     print(name, param.numel())

    # # test one batch in model 
    # batch = next(iter(train_loader))

    # x_o, y_trafo_pred = model(batch)
    # print(x_o)
    # print("\n\n")
    # print(y_trafo_pred[0])
    # print("\n\n")
    # # print(batch[0].y_tap[:,0])
    # exit()

    # checkpoint = torch.load(os.getcwd()+"/config/TapSEGNN_e500_nDFT_TNP_lr0.01_d4000_std0.3_sct3.pth", 
    #                         weights_only=True)
    
    # # restore model 
    # model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.Adam(model.parameters(),lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
        
    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, 
                                                        mode='min',
                                                        factor=0.1, 
                                                        patience=1,
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

    results = get_eval_results(test_loader=all_loaders[2],
                                tap_weight=config['training']['loss_tap_weight'], 
                            trained_model=model, 
                            scaler=sampled_input_data['scaler_y_label'])
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    generate_markdown_report_and_save_model(current_time=current_time,
                            all_losses=all_losses, 
                            config=config, 
                            results=results, 
                            plot_loader=plot_loader, 
                            trained_model=model, 
                            sampled_input_data=sampled_input_data, 
                            usetex=False,
                            save_model=config['model']['save_model'])

if __name__ == "__main__":
    main()

