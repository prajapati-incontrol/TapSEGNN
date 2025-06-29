from typing import Literal, List, Tuple, Callable
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch_geometric.loader import DataLoader
import time


def log_cosh_loss(pred, target): 
    return torch.mean(torch.log(torch.cosh(pred - target)))

# def trainer_rq6(se_model: nn.Module, 
#                 tap_model: nn.Module,
#                 train_loader: DataLoader, 
#                 val_loader: DataLoader, 
#                 test_loader: DataLoader, 
#                 optimizer: Optimizer, 
#                 schedular: LRScheduler, 
#                 tap_schedular: LRScheduler,
#                 num_se_epoch: int, 
#                 num_tap_epoch: int,
#                 device: Literal['cpu','cuda','mps'] = 'cpu') -> Tuple:
#     """
#     (TODO)
#     Training recursive state-parameter estimation. 
#     """
#     train_losses = []
#     val_losses = []
#     start_tr = time.perf_counter()
#     for epoch in range(num_se_epoch): 
#         # only SE
#         train_loss = train_epoch_se(se_model, train_loader, optimizer, device=device)
#         val_loss = eval_epoch_se(se_model, val_loader, device=device)
#         if schedular != None: 
#             schedular.step(val_loss)
#         schedular_last_lr = schedular.get_last_lr()[0]

#         train_losses.append(train_loss)
#         val_losses.append(val_loss)

#         if epoch % int(0.05 * num_se_epoch) == 0: 
#             print(f"(SE) At epoch: {epoch}, \t training loss: {train_loss:.3e}, \
#                 \t validation loss: {val_loss:.3e} \t lr: {schedular_last_lr:.2e}") 
    
#     end_tr = time.perf_counter()
#     elapsed_tr = end_tr - start_tr

#     print(f"SE Model training took {elapsed_tr:.3f} seconds.")

#     test_loss = eval_epoch_se(se_model, test_loader, device=device)
#     print(f"Test Loss of the resulting SE model is {test_loss:.8e}")

#     print("Training Tap-Predictor using Predictions from SE Model...\n")

#     test_batch = next(iter(test_loader))
#     pred_se = se_model(test_batch)

#     tap_train_losses = []
#     tap_val_losses = []
#     start_tap_m = time.perf_counter() 
#     for epoch in range(num_tap_epoch): 
#         # only Tap prediction using predictions from trained SE model 
#         tap_train_loss = train_epoch_tap(tap_model, ) 
#         tap_val_loss = eval_epoch_tap(tap_model, )
#         if tap_schedular != None: 
#             tap_schedular.step(tap_val_loss)
#         tap_schedular_last_lr = tap_schedular.get_last_lr()[0]

#         tap_train_losses.append(tap_train_loss)
#         tap_val_losses.append(tap_val_loss)

#         if epoch % int(0.05 * num_tap_epoch) == 0:
#             print(f"(Tap) At epoch: {epoch}, \t training loss: {train_loss:.3e}, \
#                   \t validation loss: {val_loss:.3e} \t lr: {tap_schedular_last_lr:.2e}")
    
#     end_tap_m =  time.perf_counter()
#     elapsed_tap_m = end_tap_m - start_tap_m 

#     print(f"Tap Prediction Model took {elapsed_tap_m:.3f} seconds.")

#     tap_test_loss = eval_epoch_tap(tap_model,)
#     print(f"Test Loss of the resulting Tap-Prediction Model is {tap_test_loss:.8e}")

#     return (train_losses, val_losses, test_loss), (tap_train_losses, tap_val_losses, tap_test_loss) 

def train_epoch_tap(tap_model: nn.Module, 
                    loader: DataLoader, 
                    optimizer: Optimizer, 
                    se_model: nn.Module, 
                    device: Literal['cpu','mps','cuda']='cpu') -> float: 
    
    tap_model.train() 
    se_model.eval()
    criterion_tap = nn.CrossEntropyLoss()

    tap_train_loss = 0. 
    for batch in loader: 
        optimizer.zero_grad()
        y_pred_tap = tap_model(batch, se_model) 
        loss_all_tap = 0. 
        for trafo_id in y_pred_tap.keys():
            single_trafo_y_pred_tap = y_pred_tap[trafo_id].to(device) # batch_size * num_classes 
            single_trafo_y_target_tap = batch[0].y_tap[:,trafo_id].to(device) # batch_size * target class 
            loss_all_tap += criterion_tap(single_trafo_y_pred_tap, single_trafo_y_target_tap) 
        loss_all_tap.backward()
        optimizer.step()
        tap_train_loss += loss_all_tap.item() 
        
    return tap_train_loss / len(loader)

def eval_epoch_tap(tap_model: nn.Module, 
                    loader: DataLoader, 
                    se_model: nn.Module,  
                    device: Literal['cpu','mps','cuda']='cpu') -> float: 
    tap_model.eval()
    se_model.eval()
    criterion_tap = nn.CrossEntropyLoss()

    tap_train_loss = 0. 
    for batch in loader: 
        y_pred_tap = tap_model(batch, se_model) 
        loss_all_tap = 0. 
        for trafo_id in y_pred_tap.keys():
            single_trafo_y_pred_tap = y_pred_tap[trafo_id].to(device) # batch_size * num_classes 
            single_trafo_y_target_tap = batch[0].y_tap[:,trafo_id].to(device) # batch_size * target class 
            loss_all_tap += criterion_tap(single_trafo_y_pred_tap, single_trafo_y_target_tap) 
        tap_train_loss += loss_all_tap.item() 
    return tap_train_loss / len(loader)

# def trainer_fcnn(model: nn.Module, 
#                 train_loader: DataLoader, 
#                 val_loader: DataLoader, 
#                 test_loader: DataLoader,
#                 optimizer: Optimizer,
#                 schedular: LRScheduler,
#                 num_epoch: int,
#                 device: Literal['cpu','cuda','mps'] = 'cpu') -> Tuple[List, List, List, List]:
#     """Returns training, validation and testing losses for FCNN."""
#     train_losses = []
#     val_losses = []
#     gradient_norms = []
#     start_tr = time.perf_counter() # time.time is not accurate
#     for epoch in range(num_epoch):
#         model.train()
#         for batch_fcnn in train_loader:
#             inputs, labels = batch_fcnn
#             optimizer.zero_grad()
#             pred_fcnn = model(inputs)
#             loss = criterion_se(pred_fcnn, labels)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item() 

def train_epoch_fcnn_se(model: nn.Module, 
          loader: DataLoader, 
          optimizer: Optimizer,
          device: Literal['cpu', 'cuda', 'mps'] = 'cpu') -> Tuple:

    model.train()

    train_loss = 0
    grad_norm_batch = 0.
    criterion = nn.MSELoss()

    for batch in loader: 
        # print(batch)
        inputs, labels = batch
        optimizer.zero_grad()
        pred_fcnn = model(inputs)
        loss = criterion(pred_fcnn, labels.to(device))
        loss.backward()

        # gradient norm 
        grad_norm = torch.norm(torch.stack([
            p.grad.norm() for p in model.parameters() if p.grad is not None
        ]))

        grad_norm_batch += grad_norm.item()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(loader), grad_norm_batch / len(loader)

def eval_epoch_fcnn_se(model: nn.Module, 
          loader: DataLoader, 
          device: Literal['cpu', 'cuda', 'mps'] = 'cpu') -> float:
    model.eval()
    val_loss = 0. 
    criterion = nn.MSELoss() 

    with torch.no_grad(): 
        for batch in loader: 
            inputs, labels = batch 
            pred_fcnn = model(inputs)
            loss = criterion(pred_fcnn, labels.to(device))
            val_loss += loss.item()

    return val_loss / len(loader)


def trainer(model: nn.Module, 
            train_loader: DataLoader, 
            val_loader: DataLoader, 
            test_loader: DataLoader,
            optimizer: Optimizer,
            schedular: LRScheduler,
            num_epoch: int,
            early_stopping: bool,
            val_patience: int,
            tap_weight: float = 1, 
            device: Literal['cpu','cuda','mps'] = 'cpu') -> Tuple[List,List,List,List]:
    """Returns training, validation and testing losses."""
    train_losses = []
    val_losses = []
    gradient_norms = []
    start_tr = time.perf_counter() # time.time is not accurate
    fcnn_nets = {"FCNNRegressor"}
    multi_tapse_nets = {"MultiTapSEGNN"}
    tapse_nets = {"TapGNN","TapNRegressor", "TapSEGNN"}
    se_nets = {"NGATRegressor","GATRegressor","NERegressor", "NRegressor", "TAGNRegressor4SE","EdgeRegressor", "EdgeLGRegressor","NEGATRegressor","NEGATRegressor_LGL"}
    early_stop = 0
    criterion_se_v = nn.MSELoss() 
    criterion_se_a = nn.L1Loss()
    angle_weight = 1.1
    for epoch in range(num_epoch):
        # if model.name in tapse_nets:
        #     train_loss, se_train_loss, tap_train_loss = train_epoch_tapse(model, train_loader, optimizer, tap_weight, device)
        #     val_loss, se_val_loss, tap_val_loss = eval_epoch_tapse(model, val_loader, tap_weight, device)
        #     if schedular != None: 
        #         schedular.step(val_loss)
        #     schedular_last_lr = schedular.get_last_lr()[0] # since output is list
        if model.name in se_nets:
            # training loss per epoch
            train_loss, gradient_norm = train_epoch_se(model, 
                                        train_loader, 
                                        optimizer, 
                                        criterion_se_v=criterion_se_v, 
                                        criterion_se_a=criterion_se_a,
                                        angle_weight=angle_weight,
                                        device=device)
            val_loss = eval_epoch_se(model,
                                     val_loader, 
                                     criterion_se_v=criterion_se_v, 
                                     criterion_se_a=criterion_se_a,
                                     angle_weight=angle_weight,
                                     device=device)
        elif model.name in multi_tapse_nets: 
            train_loss, se_train_loss, all_tap_train_loss, gradient_norm = train_epoch_multitapse(model, 
                                                                                   train_loader, 
                                                                                   optimizer, 
                                                                                   weight=tap_weight, 
                                                                                   criterion_se_v=criterion_se_v,
                                                                                   criterion_se_a=criterion_se_a,
                                                                                   angle_weight=angle_weight,
                                                                                   device=device)
            val_loss, se_val_loss, all_tap_val_loss = eval_epoch_multitapse(model, 
                                                                            val_loader, 
                                                                            weight=tap_weight,
                                                                            criterion_se_v=criterion_se_v,
                                                                            criterion_se_a=criterion_se_a,
                                                                            angle_weight=angle_weight,
                                                                            device=device)
        elif model.name in fcnn_nets:
            train_loss, gradient_norm = train_epoch_fcnn_se(model, 
                                                            train_loader, 
                                                            optimizer,
                                                            device=device)
            val_loss = eval_epoch_fcnn_se(model, 
                                          val_loader, 
                                          device=device)
        else: 
            raise NotImplementedError
        if schedular != None: 
            schedular.step(val_loss)
        schedular_last_lr = schedular.get_last_lr()[0]
        # else: 
        #     # training loss per epoch
        #     train_loss = ptr_train_epoch(model, train_loader, optimizer, schedular, device)
        #     val_loss = ptr_eval_epoch(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        gradient_norms.append(gradient_norm)

        if early_stopping:
            # early stopping if validation loss starts increasing 
            if len(val_losses) > 5: # after at least 2 epochs  
                if val_losses[-1] >= val_losses[-2]:
                    early_stop += 1
                    if early_stop >= val_patience: 
                        print(f"Early Stopping at epoch {epoch}...")
                        break
                else: 
                    early_stop = 0

                # Check if the last 10 validation losses are similar (convergence)
            if len(val_losses) >= 50:
                last_50_losses = val_losses[-50:]
                max_loss = max(last_50_losses)
                min_loss = min(last_50_losses)
                tolerance = 1e-10  # Define a small tolerance for similarity
                if (max_loss - min_loss) < tolerance:
                    print(f"Early Stopping at epoch {epoch} due to convergence (last 50 losses are similar)...")
                    break
        if epoch % max(1, int(0.05 * num_epoch)) == 0: 
            # if model.name in tapse_nets:
            #     print(f"At epoch: {epoch}, \t training loss: {se_train_loss:.3e} + {tap_weight:.1e} * {tap_train_loss:.3e} = {train_loss:.3e}, \
            #       \t validation loss: {val_loss:.3e} \t lr: {schedular_last_lr:.2e}") # {se_val_loss:.3e} + {tap_weight:.1e} * {tap_val_loss:.3e} = 
            if model.name in se_nets:
                print(f"At epoch: {epoch}, \t training loss: {train_loss:.3e}, \
                    \t validation loss: {val_loss:.3e} \t lr: {schedular_last_lr:.2e} \t grad_norm: {gradient_norm:.3e}") 
            elif model.name in multi_tapse_nets: 
                print(f"At epoch: {epoch}, \t training loss: {se_train_loss:.3e} + {tap_weight:.1e} * {all_tap_train_loss:.3e} = {train_loss:.3e}, \t validation loss: {se_val_loss:.3e} + {tap_weight:.1e} * {all_tap_val_loss:.3e} = {val_loss:.3e} \t lr: {schedular_last_lr:.2e} \t grad_norm: {gradient_norm:.3e}")
            elif model.name in fcnn_nets: 
                print(f"At epoch: {epoch}, \t training loss: {train_loss:.3e}, \
                    \t validation loss: {val_loss:.3e} \t lr: {schedular_last_lr:.2e} \t grad_norm: {gradient_norm:.3e}")
            else: 
                print(f"At epoch: {epoch}, \t training loss: {train_loss:.3e}, \
                    \t validation loss: {val_loss:.3e} \t lr: {schedular_last_lr:.2e}") 
            
    end_tr = time.perf_counter()
    elapsed_tr = end_tr - start_tr

    print(f"Model training took {elapsed_tr:.3f} seconds.")

    # if model.name in {"TapGNN","TapNRegressor", "TapSEGNN"}:
    #     test_loss = eval_epoch_tapse(model, test_loader, weight=tap_weight, device=device)[0]
    if model.name in {"NGATRegressor","GATRegressor","NERegressor", "NRegressor", "TAGNRegressor4SE","EdgeRegressor","EdgeLGRegressor","NEGATRegressor","NEGATRegressor_LGL"}:
        test_loss = eval_epoch_se(model, 
                                  test_loader, 
                                  criterion_se_v=criterion_se_v,
                                  criterion_se_a=criterion_se_a,
                                  angle_weight=angle_weight,
                                  device=device)
    elif model.name in {"MultiTapSEGNN"}:
        test_loss = eval_epoch_multitapse(model, 
                                          test_loader, 
                                          weight=tap_weight, 
                                          criterion_se_v=criterion_se_v,
                                          criterion_se_a=criterion_se_a,
                                          angle_weight=angle_weight,
                                          device=device)[0]
    elif model.name in fcnn_nets: 
        test_loss = eval_epoch_fcnn_se(model, 
                                       test_loader, 
                                       device=device)
    else: 
        raise NotImplementedError("Model not available for calculating test loss.")

    print(f"Test Loss of the resulting model is {test_loss:.8e}")

    return train_losses, val_losses, test_loss, gradient_norms



######################## TRAIN AND EVAL EPOCH FOR SE AND Multi-TAP ####################################

def train_epoch_multitapse(model: nn.Module, 
                           loader: DataLoader, 
                           optimizer: Optimizer, 
                           weight: float,
                           criterion_se_v: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
                           criterion_se_a: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                           angle_weight: float = 1.1, 
                           device: Literal['cpu','cuda','mps'] = 'cpu') -> float: 
    
    model.train()

    # i have two criterias for two model outputs, one output is for mseloss and other for tap classification
    criterion_tap = nn.CrossEntropyLoss()
    
    train_loss = 0. 
    se_train_loss = 0.
    tap_train_loss = 0. 
    grad_norm_batch = 0.
    for batch in loader: 
        optimizer.zero_grad()
        y_pred_se, y_pred_tap = model(batch)
        # loss_se = criterion_se(y_pred_se, batch[0].y.to(device))
        loss_se_v = criterion_se_v(y_pred_se[:,0], batch[0].y[:,0].to(device))
        loss_se_a = criterion_se_a(y_pred_se[:,1], batch[0].y[:,1].to(device))
        loss_se = loss_se_v + angle_weight * loss_se_a 
        loss_all_tap = 0. 
        for trafo_id in y_pred_tap.keys():
            single_trafo_y_pred_tap = y_pred_tap[trafo_id].to(device) # batch_size * num_classes 
            single_trafo_y_target_tap = batch[0].y_tap[:,trafo_id].to(device) # batch_size * target class 
            loss_all_tap += criterion_tap(single_trafo_y_pred_tap, single_trafo_y_target_tap) 
        loss = loss_se + weight * loss_all_tap 
        loss.backward()
        
        # gradient norm 
        grad_norm = torch.norm(torch.stack([
            p.grad.norm() for p in model.parameters() if p.grad is not None
        ]))

        grad_norm_batch += grad_norm.item()
        optimizer.step()
        train_loss += loss.item()
        se_train_loss += loss_se.item()
        tap_train_loss += loss_all_tap.item() 
        
    
    return train_loss / len(loader), se_train_loss / len(loader), tap_train_loss / len(loader), grad_norm_batch/len(loader)

def eval_epoch_multitapse(model:nn.Module, 
                     loader: DataLoader,
                     weight: float,
                     criterion_se_v: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
                     criterion_se_a: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                     angle_weight: float = 1.1, 
                     device: Literal['cpu', 'cuda', 'mps'] = 'cpu') -> float: 
    model.eval()

    criterion_tap = nn.CrossEntropyLoss()

    with torch.no_grad(): 
        val_loss = 0.
        se_val_loss = 0.
        all_tap_val_loss = 0.
        for batch in loader: 
            y_pred_se_val, y_pred_tap_val = model(batch)
            # loss_se = criterion_se(y_pred_se_val, batch[0].y.to(device))
            loss_se_v = criterion_se_v(y_pred_se_val[:,0], batch[0].y[:,0].to(device))
            loss_se_a = criterion_se_a(y_pred_se_val[:,1], batch[0].y[:,1].to(device))
            loss_se = loss_se_v + loss_se_a
            for trafo_id in y_pred_tap_val.keys():
                single_trafo_y_pred_tap = y_pred_tap_val[trafo_id].to(device) # batch_size * num_classes 
                single_trafo_y_target_tap = batch[0].y_tap[:,trafo_id].to(device) # batch_size * target class 
                all_tap_val_loss += criterion_tap(single_trafo_y_pred_tap, single_trafo_y_target_tap) 
            loss = loss_se + weight * all_tap_val_loss
            val_loss += loss.item()
            se_val_loss += loss_se.item()
            all_tap_val_loss += all_tap_val_loss.item()
    return val_loss / len(loader), se_val_loss / len(loader), all_tap_val_loss / len(loader)

######################## TRAIN AND EVAL EPOCH FOR SE ONLY #######################################

def train_epoch_se(model: nn.Module, 
          loader: DataLoader, 
          optimizer: Optimizer,
          criterion_se_v: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
          criterion_se_a: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
          angle_weight: float = 1.1,
          device: Literal['cpu', 'cuda', 'mps'] = 'cpu') -> float:

    model.train()

    train_loss = 0
    train_loss_ref = 0
    grad_norm_batch = 0.

    for batch in loader: 
        optimizer.zero_grad()
        y_pred = model(batch)
        # y_label = batch[0].y.to(device)
        loss_se_v = criterion_se_v(y_pred[:,0], batch[0].y[:,0].to(device))
        loss_se_a = criterion_se_a(y_pred[:,1], batch[0].y[:,1].to(device))
        loss = loss_se_v + angle_weight*loss_se_a 
        loss_ref = loss_se_v + loss_se_a
        # loss = criterion(y_pred, y_label)
        loss.backward()

        # gradient norm 
        grad_norm = torch.norm(torch.stack([
            p.grad.norm() for p in model.parameters() if p.grad is not None
        ]))

        grad_norm_batch += grad_norm.item()
        optimizer.step()
        train_loss += loss.item()
        train_loss_ref += loss_ref.item()

    return train_loss_ref / len(loader), grad_norm_batch / len(loader)

def eval_epoch_se(model: nn.Module, 
             loader: DataLoader,
             criterion_se_v: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
             criterion_se_a: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
             angle_weight: float = 1.1, 
             device: Literal['cpu', 'cuda', 'mps'] = 'cpu') -> float:
    
    model.eval()
    
    with torch.no_grad():
        val_loss = 0.
        val_loss_ref = 0.
        for batch in loader: 
            y_pred_val = model(batch)
            # y_label_val = batch[0].y.to(device)
            loss_se_v = criterion_se_v(y_pred_val[:,0], batch[0].y[:,0].to(device))
            loss_se_a = criterion_se_a(y_pred_val[:,1], batch[0].y[:,1].to(device))
            loss = loss_se_v + angle_weight * loss_se_a 
            loss_ref = loss_se_v + loss_se_a 
            # loss = criterion(y_pred_val, y_label_val)
            val_loss += loss.item()
            val_loss_ref += loss_ref.item()
    
    return val_loss_ref / len(loader)



######################## TRAIN AND EVAL EPOCH FOR PRETRAINING #######################################


# def ptr_train_epoch(model: nn.Module, 
#           loader: DataLoader, 
#           optimizer: Optimizer,
#           schedular: LRScheduler, 
#           device: Literal['cpu', 'cuda', 'mps'] = 'cpu') -> float:
    
#     model.train()

#     criterion = nn.MSELoss()

#     train_loss = 0
#     for batch in loader: 
#         optimizer.zero_grad()
#         y_pred = model(batch).to(device)
#         y_label = batch.x
#         loss = criterion(y_pred, y_label)
#         loss.backward()
#         optimizer.step()
#         if schedular != None: 
#             schedular.step()
#         train_loss += loss.item()

#     return train_loss / len(loader)

# def ptr_eval_epoch(model: nn.Module, 
#              loader: DataLoader, 
#              device: Literal['cpu', 'cuda', 'mps'] = 'cpu') -> float:
    
#     model.eval()
    
#     criterion = nn.MSELoss()

#     with torch.no_grad():
#         val_loss = 0.
#         for batch in loader: 
#             y_pred_val = model(batch).to(device)
#             loss = criterion(y_pred_val, batch.x)
#             val_loss += loss.item()
    
#     return val_loss / len(loader)


######################## TRAIN AND EVAL EPOCH FOR SE AND TAP ####################################

# def train_epoch_tapse(model: nn.Module, 
#                       loader: DataLoader, 
#                       optimizer: Optimizer, 
#                       weight: float,
#                       criterion_se_v: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
#                       criterion_se_a: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
#                       device: Literal['cpu','cuda','mps'] = 'cpu') -> float: 
#     model.train()

#     # i have two criterias for two model outputs, one output is for mseloss and other for tap classification
#     criterion_tap = nn.CrossEntropyLoss()
    
#     train_loss = 0. 
#     se_train_loss = 0.
#     tap_train_loss = 0. 
#     for batch in loader: 
#         optimizer.zero_grad()
#         # print(batch)
#         y_pred_se, y_pred_tap = model(batch)
#         y_pred_se, y_pred_tap = y_pred_se.to(device), y_pred_tap.to(device)
#         loss_se = criterion_se(y_pred_se, batch[0].y.to(device))
#         loss_tap = criterion_tap(y_pred_tap, batch[0].y_tap.to(device)) # pred, target
#         loss = loss_se + weight * loss_tap 
#         # loss = loss_tap
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#         se_train_loss += loss_se.item()
#         tap_train_loss += loss_tap.item() 
        
    
#     return train_loss / len(loader), se_train_loss / len(loader), tap_train_loss / len(loader)

# def eval_epoch_tapse(model:nn.Module, 
#                      loader: DataLoader,
#                      weight: float, 
#                      device: Literal['cpu', 'cuda', 'mps'] = 'cpu') -> float: 
#     model.eval()
    
#     criterion_se = nn.MSELoss()
#     criterion_tap = nn.CrossEntropyLoss()

#     with torch.no_grad(): 
#         val_loss = 0.
#         se_val_loss = 0.
#         tap_val_loss = 0.
#         for batch in loader: 
#             y_pred_se_val, y_pred_tap_val = model(batch)
#             y_pred_se_val, y_pred_tap_val = y_pred_se_val.to(device), y_pred_tap_val.to(device)
#             loss_se = criterion_se(y_pred_se_val, batch[0].y.to(device))
#             loss_tap = criterion_tap(y_pred_tap_val, batch[0].y_tap.to(device)) # pred, target
#             loss = loss_se + weight * loss_tap
#             # loss = loss_tap
#             val_loss += loss.item()
#             se_val_loss += loss_se.item()
#             tap_val_loss += loss_tap.item()
#     return val_loss / len(loader), se_val_loss / len(loader), tap_val_loss / len(loader)
