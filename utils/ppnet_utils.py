import networkx as nx 
import numpy as np
import pandapower as pp
import warnings 
import torch
import copy
import os
import sys 
import pickle
import pandas as pd 
from typing import Tuple, List, Dict
from torch_geometric.data import Data, Batch, Dataset
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import pylab
import igraph as ig
import time 

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

##########################################################################################################

def initialize_network(net_name: str,
                       else_load: float = 0.0001,  
                       verbose: bool = True) -> pp.pandapowerNet: 
    
    # else_load = 1.0

    match net_name: 
        case 'PP_MV_RING':
            net = pp.networks.simple_pandapower_test_networks.simple_mv_open_ring_net()
            net.load.loc[:, 'p_std'] = else_load
            net.trafo.shift_degree = 0.0 # bug in pandapower, initialized trafo shift degree is 150 degrees.
            if verbose: 
                nonhv_trafo_ids = ~net.trafo.loc[:,"name"].str.contains("HV", na=False)
                mvlv_trafo_ids = list(net.trafo.loc[nonhv_trafo_ids].index)
                min_mvlv = min(mvlv_trafo_ids)
                max_mvlv = max(mvlv_trafo_ids)
                print(f"Transformer Indices for {net_name} are available from [{min_mvlv},{max_mvlv}] \n")
                print(f"Number of Trafos = {len(net.trafo)} \n ")
        
        case 'TOY':
            # load toy network
            net = pp.from_pickle(parent_dir + '/data/net_TOY.p')
            net.load.loc[:, 'p_std'] = else_load
            net.trafo.shift_degree = 0.0
            net.name = net_name 
            if verbose: 
                nonhv_trafo_ids = ~net.trafo.loc[:,"name"].str.contains("HV", na=False)
                mvlv_trafo_ids = list(net.trafo.loc[nonhv_trafo_ids].index)
                min_mvlv = min(mvlv_trafo_ids)
                max_mvlv = max(mvlv_trafo_ids)
                print(f"Transformer Indices for {net_name} are available from [{min_mvlv},{max_mvlv}] \n")
                print(f"Number of Trafos = {len(net.trafo)} \n ")
            
        case 'MVO':
            net = pp.networks.mv_oberrhein(include_substations=True)
            net.load.loc[:, 'p_std'] = else_load
            net.trafo.loc[net.trafo.loc[:,'tap_step_percent'].isna(),'tap_step_percent'] = 2.173913
            net.name = net_name
            if verbose: 
                nonhv_trafo_ids = ~net.trafo.loc[:,"name"].str.contains("HV", na=False)
                mvlv_trafo_ids = list(net.trafo.loc[nonhv_trafo_ids].index)
                min_mvlv = min(mvlv_trafo_ids)
                max_mvlv = max(mvlv_trafo_ids)
                print(f"Transformer Indices for {net_name} are available from [{min_mvlv},{max_mvlv}] \n")
                print(f"Number of Trafos = {len(net.trafo)} \n ")
    
        case _:
            raise NameError("\n Invalid Network Name! ")
        
    print(f"Network: {net_name} is selected \n")
    netG = pp.topology.create_nxgraph(net)
    print(f"Net {net_name} has {len(netG.nodes())} nodes and {len(netG.edges())} edges. \n")

    return net 


#####################################################################################

def get_trafo_ids_from_percent(net: pp.pandapowerNet, trafo_id_percent: int): 

    # for consistent results 
    nonhv_trafo_ids = ~net.trafo.loc[:,"name"].str.contains("HV", na=False)
    mvlv_trafo_ids = list(net.trafo.loc[nonhv_trafo_ids].index)

    n_mvlv_trafos = len(mvlv_trafo_ids)

    match trafo_id_percent: 
        
        case 0: # no transformer only SE 
            return []

        case 1: # single transformer 
            return [mvlv_trafo_ids[0]]
        
        case 25: 
            end_idx = n_mvlv_trafos // 4 
            return mvlv_trafo_ids[:end_idx]
        
        case 50: 
            end_idx = n_mvlv_trafos // 2
            return mvlv_trafo_ids[:end_idx]
        
        case 75: 
            end_idx = (3 * n_mvlv_trafos) // 4
            return mvlv_trafo_ids[:end_idx]

        case 100: 
            return ["all"]


#####################################################################################

def add_branch_parameters(net: pp.pandapowerNet): 
    """Calculates the branch parameters of the bus-branch model. 
    if branch is line, then net.line has added columns as r, x, b, g
    if branch is trafo, then net.trafo has added columns as r, x, b, g, tap, shift 
    
    Assumptions: 
    No mutual inductance between lines. 

    Note: 
    Pandapower equations from documentation for trafo r, x, b and g are ambiguous, and do not match with the 
    internal y-bus matrices. So, branch parameters for trafo are calculated from Y_bus internals and not 
    the pandapower equations. 

    Returns: 
    net: Pandapower Net with added branch parameter for line and trafo. 
    
    """
    # remove switches 
    net.switch.drop(net.switch.index, inplace = True)
    net.res_switch.drop(net.res_switch.index, inplace = True)

    sys_freq = 50 # [Hz]
    
    # check 1: if the required columns are present in lines dataframe  
    required_columns = {
        'line':['r_ohm_per_km', 'length_km', 'parallel', 'from_bus', 'x_ohm_per_km', 'c_nf_per_km', 'g_us_per_km'],
        'bus':['vn_kv'],
        'trafo': ['lv_bus', 'vn_lv_kv', 'sn_mva', 'vkr_percent', 'vk_percent', 'pfe_kw', 'i0_percent', 'hv_bus', 'vn_hv_kv']
    }

    for element, columns in required_columns.items(): 
        for column in columns:
            if column not in net[element].columns:
                warnings.warn(f"Column '{column}' is missing in '{element}', padding with zeros.")
                net[element][column] = 0    
    
    # line branch parameters 
    sys_freq = 50

    # convert r_ohm_per_km to r_pu
    r_pu = (net.line['r_ohm_per_km'] * net.line['length_km'] / net.line['parallel']) / (net.bus['vn_kv'].loc[net.line['from_bus'].values]**2/net.sn_mva).values
    net.line.loc[:,'r_pu'] = r_pu

    # convert x_ohm_per_km to x_ohm
    x_pu = (net.line['x_ohm_per_km'] * net.line['length_km'] / net.line['parallel']) / (net.bus['vn_kv'].loc[net.line['from_bus'].values]**2/net.sn_mva).values
    net.line.loc[:,'x_pu'] = x_pu

    z_pu = r_pu + 1j*x_pu
    y_series = 1 / z_pu

    # convert c_nf_per_km to b_mho
    b_pu = ( 2 * np.pi * sys_freq * net.line['c_nf_per_km'] * 10**(-9) * net.line['length_km'] * net.line['parallel']) * (net.bus['vn_kv'].loc[net.line['from_bus'].values]**2/net.sn_mva).values 
    net.line.loc[:, 'b_pu'] = b_pu

    # convert g_us_per_km to g_mho
    g_pu = ( 2 * np.pi * sys_freq * net.line['g_us_per_km'] * 10**(-6) * net.line['length_km'] * net.line['parallel']) * (net.bus['vn_kv'].loc[net.line['from_bus'].values]**2/net.sn_mva).values
    net.line.loc[:, 'g_pu'] = g_pu

    # add zeros for tap and shift degree in lines 
    net.line.loc[:, 'tap_nom'] = 0 
    net.line.loc[:, 'shift_rad'] = 0


    y_sh = g_pu - 1j*b_pu    

    # get Y-bus for lines only 
    a_1 = y_series + y_sh/2
    a_2 = - y_series
    a_3 = - y_series
    a_4 = y_series + y_sh/2

    yb_size = len(net.bus)

    Ybline = np.zeros((yb_size, yb_size)).astype(complex)

    fb_line = net.line.from_bus 
    tb_line = net.line.to_bus 

    line_idx = net.line.index 

    for (idx, fb, tb) in zip(line_idx, fb_line, tb_line): 
        Ybline[fb, fb] = complex(a_1[idx])
        Ybline[fb, tb] = complex(a_2[idx])
        Ybline[tb, fb] = complex(a_3[idx])
        Ybline[tb, tb] = complex(a_4[idx])

    # get the pandpaower internal YBus 
    pp.runpp(net)
    Ybus = np.array(net._ppc["internal"]["Ybus"].todense())

    Ybus_trafo = Ybus - Ybline 

    if sum(net.trafo.tap_pos.isna()) > 0: 
        print("Filling nan as 0 in tap_pos, tap_neutral, tap_step_degree")
        net.trafo.loc[net.trafo.loc[:,'tap_pos'].isna(),'tap_pos'] = 0
        net.trafo.loc[net.trafo.loc[:,'tap_neutral'].isna(),'tap_neutral'] = 0
        net.trafo.loc[net.trafo.loc[:,'tap_step_degree'].isna(),'tap_step_degree'] = 0
        
        

    ps_s = net.trafo.shift_degree * np.pi/180
    net.trafo.loc[:,"shift_rad"] = ps_s
    tap_nom_s = 1 + (net.trafo.tap_pos - net.trafo.tap_neutral) * (net.trafo.tap_step_percent / 100)
    net.trafo.loc[:,"tap_nom"] = tap_nom_s 
    N_tap_s = tap_nom_s * np.exp(1j*ps_s)


    for id, row in net.trafo.iterrows(): 
        a_1t = Ybus_trafo[row.hv_bus, row.hv_bus]
        a_2t = Ybus_trafo[row.hv_bus, row.lv_bus]
        a_3t = Ybus_trafo[row.lv_bus, row.hv_bus]
        a_4t = Ybus_trafo[row.lv_bus, row.lv_bus]

        # series impedance 
        y_series_trafo = - a_3t * N_tap_s[id] 

        # r, x 
        z_s_trafo = 1 / y_series_trafo
        r_trafo, x_trafo = np.real(z_s_trafo), np.imag(z_s_trafo)

        net.trafo.loc[id, 'r_pu'] = r_trafo
        net.trafo.loc[id, 'x_pu'] = x_trafo 

        # shunt impedance 
        y_sh_trafo = 2* (a_4t - y_series_trafo) 
        g_trafo, b_trafo = np.real(y_sh_trafo), np.imag(y_sh_trafo)

        net.trafo.loc[id,'g_pu'] = g_trafo 
        net.trafo.loc[id,'b_pu'] = b_trafo
    
    return net

def drop_hv_trafos(net:pp.pandapowerNet):
    """
    NOTE: Always call this function after add_branch_parameters(net_name)
    NOTE: Never run power flow after calling this function.
    
    Drop Trafo rows for HV trafos. Useful for analysing MV/LV Trafo parameters.
    
    """
    
    # get HV indices.
    net.trafo.drop(index=net.trafo.index[net.trafo.loc[:,"name"].str.contains("HV", na=False)], inplace=True)
    net.trafo.reset_index(inplace=True)

    return net


#####################################################################################

def drop_pf_results(net:pp.pandapowerNet):
    """
    Drop any 'result' datastructure if any.
    """
    net.res_bus.drop(net.res_bus.index, inplace=True)
    net.res_line.drop(net.res_line.index, inplace=True)
    net.res_load.drop(net.res_load.index, inplace=True)
    net.res_trafo.drop(net.res_trafo.index, inplace=True)
    net.res_ext_grid.drop(net.res_ext_grid.index, inplace=True)
    net.res_sgen.drop(net.res_sgen.index, inplace=True)
    net.res_switch.drop(net.res_switch.index, inplace=True)
    return net 

def use_stored_pfr(net: pp.pandapowerNet, parent_dir):
    """Use stored power flow results."""
    # parent_dir = os.getcwd()
    try: 
        vm_pfr = pickle.load(open(parent_dir+ f"/data/{net.name}/{net.name}" + '_vm_pfr.pkl', 'rb'))
        va_pfr = pickle.load(open(parent_dir+ f"/data/{net.name}/{net.name}" + '_va_pfr.pkl', 'rb'))
        p_pfr = pickle.load(open(parent_dir+ f"/data/{net.name}/{net.name}" + '_p_pfr.pkl', 'rb'))
        q_pfr = pickle.load(open(parent_dir+ f"/data/{net.name}/{net.name}" + '_q_pfr.pkl', 'rb'))
        p_pfr_line = pickle.load(open(parent_dir+ f"/data/{net.name}/{net.name}" + '_p_pfr_line.pkl', 'rb'))
        print("Loaded pkls")
    except: 
        pp.runpp(net)
        vm_pfr = net.res_bus.vm_pu
        va_pfr = net.res_bus.va_degree
        p_pfr = net.res_bus.p_mw 
        q_pfr = net.res_bus.q_mvar
        p_pfr_line = net.res_line.p_from_mw
        os.makedirs(parent_dir + f"/data/{net.name}/", exist_ok=True)
        pickle.dump(vm_pfr, open(parent_dir+ f"/data/{net.name}/{net.name}" + '_vm_pfr.pkl', 'wb'))
        pickle.dump(va_pfr, open(parent_dir+ f"/data/{net.name}/{net.name}" + '_va_pfr.pkl', 'wb'))
        pickle.dump(p_pfr, open(parent_dir+ f"/data/{net.name}/{net.name}" + '_p_pfr.pkl', 'wb'))
        pickle.dump(q_pfr, open(parent_dir+ f"/data/{net.name}/{net.name}" + '_q_pfr.pkl', 'wb'))
        pickle.dump(p_pfr_line, open(parent_dir+ f"/data/{net.name}/{net.name}" + '_p_pfr_line.pkl', 'wb'))
        print("Dumped and loaded pkls")
    
    vm_pfr = np.array(vm_pfr)
    va_pfr = np.array(va_pfr)
    p_pfr = np.array(p_pfr)
    q_pfr = np.array(q_pfr)
    p_pfr_line = np.array(p_pfr_line)
    return vm_pfr, va_pfr, p_pfr, q_pfr, p_pfr_line

#####################################################################################


def custom_se(net: pp.pandapowerNet, 
              pfr_dict: Dict, 
            #   verbose: bool = False,
              se_iter: int = 2,
              is_line_meas:bool=False,
              is_for_rq2:bool=False, 
              prob_lq_meas:float=0.0) -> pp.pandapowerNet:
    """
    Custom state-estimation for a pandapower network targetted to get consistent results by using 
    a measurement vector. 
    """
    v_pfr = np.array(pfr_dict['vm_pfr'])
    va_pfr = np.array(pfr_dict['va_pfr'])
    p_pfr = np.array(pfr_dict['p_pfr'])
    q_pfr = np.array(pfr_dict['q_pfr'])
    p_pfr_line = np.array(pfr_dict['p_pfr_line'])
    
    # drop the measurements or power flow results if any 
    net.measurement.drop(net.measurement.index, inplace = True)
    net = drop_pf_results(net)

    # number of bus indices 
    n_bus = len(net.bus.index)

    v_meas_bus_idx = list(net.bus.index)
    p_meas_bus_idx = list(net.bus.index)
    q_meas_bus_idx = list(net.bus.index)
    p_meas_line_idx = list(net.line.index)
    
    # add them as measurements with uncertainty to pandapower 
    # standard deviations for v measurements are usually 1% and that of p are 5%
    a = 1
    v_std = 0.1*0.5/100/3 * a
    p_std = 5/100/3 * a
    p_line_std = 5/100/3 * a 

    # because dataframe indices is not in natural sequence
    for idx, (idx_v, idx_p) in enumerate(zip(v_meas_bus_idx, p_meas_bus_idx)): 
        vm_at_idx_v = v_pfr[idx] + np.random.normal(0, v_std)
        # relative std
        p_at_idx_p = p_pfr[idx] + np.random.normal(0,p_std) * p_pfr[idx]   
        pp.create.create_measurement(net, "v", "bus", value=vm_at_idx_v, std_dev = v_std, element = idx_v)
        pp.create.create_measurement(net, "p", "bus", value=p_at_idx_p, std_dev = p_std, element = idx_p)
    
    # no. of measurments
    n_meas = len(v_meas_bus_idx) + len(p_meas_bus_idx)

    if is_for_rq2:
        # which lines to include for creating measurement
        meas_bool_line = np.random.rand(len(p_meas_line_idx)) < (prob_lq_meas) 

        # create only those measurements 
        p_meas_line_idx = [i for idx, i in enumerate(p_meas_line_idx) if meas_bool_line[idx]]
        n_meas += len(p_meas_line_idx)
        print(f"Total line measurements = {len(p_meas_line_idx)}")
        for idx, idx_p_line in enumerate(p_meas_line_idx):
                p_meas_at_line = p_pfr_line[idx] + np.random.normal(0, p_line_std)
                pp.create.create_measurement(net, "p","line",value=p_meas_at_line, std_dev=p_line_std, side='from', element=idx_p_line)

        # which buses to include for creating measurement for reactive power 
        meas_bool_bus = np.random.rand(len(q_meas_bus_idx)) < (prob_lq_meas)

        q_meas_bus_idx = [i for idx, i in enumerate(meas_bool_bus) if meas_bool_bus[idx]]
        n_meas += len(q_meas_bus_idx)
        print(f"Reactive power measurements = {len(q_meas_bus_idx)}")
        for idx, idx_q in enumerate(q_meas_bus_idx):
            q_at_idx_q = q_pfr[idx] + np.random.normal(0, p_std) * q_pfr[idx]
            pp.create.create_measurement(net, "q", "bus", value=q_at_idx_q, std_dev = p_std, element= idx_q)
        print(f"Redundancy = {n_meas / (2*n_bus - 1)}.")
        net.eta = n_meas / (2*n_bus - 1)


    else:  
        if is_line_meas: 
            n_meas += len(p_meas_line_idx)
            for idx, idx_p_line in enumerate(p_meas_line_idx):
                p_meas_at_line = p_pfr_line[idx] + np.random.normal(0, p_line_std)
                pp.create.create_measurement(net, "p","line",value=p_meas_at_line, std_dev=p_line_std, side='from', element=idx_p_line)
        
    # drop the pf results 
    net = drop_pf_results(net)
    
    success = pp.estimation.estimate(net, algorithm='wls', init='flat') 
    
    if success: 
        print(f"State estimation successful for {net.name}!")
        # number of measurements 
        print(f"Total number of measurements = {n_meas} and number of measurements should be at least {2*n_bus - 1}")
        # print(net)
        return net
    elif not success and se_iter > 1:
        # if verbose:
        print(f"State estimation failed. Retrying... Remaining attempts: {se_iter - 1}")
        return custom_se(net=net, 
                        pfr_dict=pfr_dict,
                        is_line_meas=is_line_meas, 
                        se_iter=se_iter - 1)
    else: 
        UserWarning(f"Solver failed for {net.name}, returning net as it is.")
        # number of measurements 
        print(f"Total number of measurements = {n_meas} and number of measurements should be at least {2*n_bus - 1}")
        return net

#####################################################################################


def abcd_net(net:pp.pandapowerNet, 
             is_shift_degree: bool = False):
    """Calculates the generalized circuit constants used directly in the making 
     of the Ybus matrix. 
     Assuming no mutual inductance between lines. 

     """

    sys_freq = 50 # [Hz]

    if not is_shift_degree: 
        net.trafo.shift_degree = 0
    
    # columns required to calculate the per-unit values 
    required_columns = {
        'line':['r_ohm_per_km', 'length_km', 'parallel', 'from_bus', 'x_ohm_per_km', 'c_nf_per_km', 'g_us_per_km'],
        'bus':['vn_kv'],
        'trafo': ['lv_bus', 'vn_lv_kv', 'sn_mva', 'vkr_percent', 'vk_percent', 'pfe_kw', 'i0_percent', 'hv_bus', 'vn_hv_kv']
    }

    for element, columns in required_columns.items(): 
        for column in columns:
            if column not in net[element].columns:
                warnings.warn(f"Column '{column}' is missing in '{element}', padding with zeros.")
                net[element][column] = 0    

    # for lines
    # convert r_ohm_per_km to r_ohm
    r_ohm = (net.line['r_ohm_per_km'] * net.line['length_km'] / net.line['parallel']) 

    # convert x_ohm_per_km to x_ohm
    x_ohm = (net.line['x_ohm_per_km'] * net.line['length_km'] / net.line['parallel']) 

    z_ohm = r_ohm + 1j*x_ohm
    y_series = 1 / z_ohm
    
    # convert c_nf_per_km to b_mho
    b_mho = ( 2 * np.pi * sys_freq * net.line['c_nf_per_km'] * 10**(-9) * net.line['length_km'] * net.line['parallel']) 

    # convert g_us_per_km to g_mho
    g_mho = ( net.line['g_us_per_km'] * 10**(-6) * net.line['length_km'] * net.line['parallel']) 

    y_sh = g_mho - 1j*b_mho
    print(y_sh)

    net.line['a_1'] = y_series + y_sh/2
    net.line['a_2'] = - y_series
    net.line['a_3'] = - y_series
    net.line['a_4'] = y_series + y_sh/2

    # for transformers

    # get transformer impedance and magnetising admittance
 

    # 1. series losses: joule or heat losses
    # short-circuit resistance r
    r_s_trafo = (net.trafo['vkr_percent']/100 * net.sn_mva / net.trafo['sn_mva']) 

    # short-circuit impedance magnitude
    z_s_trafo = (net.trafo['vk_percent']/100 * net.sn_mva / net.trafo['sn_mva'])

    # short-circuit reactance 
    x_s_trafo = np.sqrt(z_s_trafo**2 - r_s_trafo**2)    

    # series impedance 
    z_s_trafo_j = r_s_trafo + 1j*x_s_trafo

    print(z_s_trafo_j)

    # series admittance 
    y_s_trafo = 1 / z_s_trafo_j

    # 2. magnetising losses: eddy currents and hysterisis 
    # magnetising conductance g
    g_sh_trafo = (net.trafo['pfe_kw']/(net.trafo['sn_mva'] * 1000) * net.sn_mva / net.trafo['sn_mva']) 

    # shunt admittance 
    y_sh_trafo = (net.trafo['i0_percent']/100) * (net.sn_mva / net.trafo['sn_mva']) 


    # magnetising impedance b
    b_sh_trafo = np.sqrt(y_sh_trafo**2- g_sh_trafo**2)

    y_sh_trafo_j = g_sh_trafo + 1j*b_sh_trafo

    if is_shift_degree: 
        phase_shift = net.trafo.shift_degree * np.pi/180
    else: 
        phase_shift = 0.0

    if sum(net.trafo.tap_pos.isna()) > 0: 
            print("Filling nan as 0 in tap_pos")
            net.trafo.tap_pos.fill_na(0, inplace=True)

    # pandapower formula
    tap_nom = 1 + (net.trafo.tap_pos - net.trafo.tap_neutral) * (net.trafo.tap_step_percent / 100)
    N_tap = tap_nom * np.exp(1j*phase_shift)
    
    # pi model
    net.trafo['a_1'] = (y_s_trafo + y_sh_trafo_j/2) * 1 / (N_tap * np.conj(N_tap))
    net.trafo['a_2'] = -y_s_trafo * 1 / (np.conj(N_tap))
    net.trafo['a_3'] = -y_s_trafo * 1 / (N_tap)
    net.trafo['a_4'] = y_s_trafo + y_sh_trafo_j/2
            
    return net

def get_bus_geodata_pos(net: pp.pandapowerNet) -> dict:
    """Obtain the position for networkx plots from pandapower network."""
    # Create graph
    edges = list(zip(net.line.from_bus.values, net.line.to_bus.values)) + list(zip(net.trafo.hv_bus, net.trafo.lv_bus))
    g = ig.Graph(edges=edges, directed=False)

    # Generate layout
    layout = g.layout("kk")  # Fruchterman-Reingold, or try "kk", "circle", etc.

    # Convert layout to coordinates
    bus_geodata = {i: layout[i] for i in range(len(layout))}
    net.bus_geodata = pd.DataFrame.from_dict(bus_geodata, orient="index", columns=["x", "y"])

    # # Line geodata (straight lines)
    # line_geodata = {}
    # for idx, line in net.line.iterrows():
    #     from_coord = bus_geodata[line.from_bus]
    #     to_coord = bus_geodata[line.to_bus]
    #     line_geodata[idx] = [from_coord, to_coord]

    # net.line_geodata = pd.DataFrame.from_dict(line_geodata, orient='index', columns=["x","y"])

    # Create pos dictionary for NetworkX
    pos = {bus: (float(net.bus_geodata.at[bus, 'x']), float(net.bus_geodata.at[bus, 'y'])) for bus in net.bus.index}

    return pos
 
############################################################################################################################################
############################################################################################################################################

def get_positive_power_flow(net):
    p_from = net.res_line.p_from_mw 
    p_to = net.res_line.p_to_mw 
    p_hv = net.res_trafo.p_hv_mw 
    p_lv = net.res_trafo.p_lv_mw 

    p_positive_line = np.where(p_from > 0, p_from, p_to)
    p_positive_trafo = np.where(p_hv > 0, p_hv, p_lv)

    return np.concatenate([p_positive_line, p_positive_trafo])

############################################################################################################################################
############################################################################################################################################

def get_power_flow_edge_index(net):
    """
    Create directed edge index based on actual power flow direction.
    Returns PyTorch tensor [2, num_edges] where edges point in direction of power flow.
    """
    # Process lines
    p_from_line = net.res_line.p_from_mw.values
    from_bus_line = net.line.from_bus.values
    to_bus_line = net.line.to_bus.values
    
    # Use numpy.where for vectorized conditional selection
    line_from = np.where(p_from_line > 0, from_bus_line, to_bus_line)
    line_to = np.where(p_from_line > 0, to_bus_line, from_bus_line)
    
    # Process transformers
    p_hv_trafo = net.res_trafo.p_hv_mw.values
    hv_bus_trafo = net.trafo.hv_bus.values
    lv_bus_trafo = net.trafo.lv_bus.values
    
    trafo_from = np.where(p_hv_trafo > 0, hv_bus_trafo, lv_bus_trafo)
    trafo_to = np.where(p_hv_trafo > 0, lv_bus_trafo, hv_bus_trafo)
    
    # Concatenate and convert to torch tensor in one step
    from_buses = np.concatenate([line_from, trafo_from])
    to_buses = np.concatenate([line_to, trafo_to])
    
    return torch.stack([
        torch.from_numpy(from_buses).long(),
        torch.from_numpy(to_buses).long()
    ])



# def rxbgts_pu(net:pp.pandapowerNet): 

#     sys_freq = 50 # [Hz]
    
#     # columns required to calculate the per-unit values 
#     required_columns = {
#         'line':['r_ohm_per_km', 'length_km', 'parallel', 'from_bus', 'x_ohm_per_km', 'c_nf_per_km', 'g_us_per_km'],
#         'bus':['vn_kv'],
#         'trafo': ['lv_bus', 'vn_lv_kv', 'sn_mva', 'vkr_percent', 'vk_percent', 'pfe_kw', 'i0_percent', 'hv_bus', 'vn_hv_kv']
#     }

#     for element, columns in required_columns.items(): 
#         for column in columns:
#             if column not in net[element].columns:
#                 warnings.warn(f"Column '{column}' is missing in '{element}', padding with zeros.")
#                 net[element][column] = 0
    
#     # for lines
#     # convert r_ohm_per_km to r_per_unit
#     net.line['r_per_unit'] = (net.line['r_ohm_per_km'] * net.line['length_km'] / net.line['parallel']) / (net.bus['vn_kv'].loc[net.line['from_bus'].values]**2/net.sn_mva).values

#     # convert x_ohm_per_km to x_per_unit
#     net.line['x_per_unit'] = (net.line['x_ohm_per_km'] * net.line['length_km'] / net.line['parallel']) / (net.bus['vn_kv'].loc[net.line['from_bus'].values]**2/net.sn_mva).values

#     # convert c_nf_per_km to c_per_unit
#     net.line['b_per_unit'] = ( 2 * np.pi * sys_freq * net.line['c_nf_per_km'] * 10**(-9) * net.line['length_km'] * net.line['parallel']) * (net.bus['vn_kv'].loc[net.line['from_bus'].values]**2/net.sn_mva).values

#     # convert g_us_per_km to g_us_per_unit
#     net.line['g_per_unit'] = ( 2 * np.pi * sys_freq * net.line['g_us_per_km'] * 10**(-6) * net.line['length_km'] * net.line['parallel']) * (net.bus['vn_kv'].loc[net.line['from_bus'].values]**2/net.sn_mva).values

#     # for transformers

#     # get transformer impedance and magnetising admittance
#     # get transformer impedance and magnetising admittance
#     Z_N = pd.Series(
#     net.bus['vn_kv'].loc[net.trafo['lv_bus'].values].values**2 / net.sn_mva,
#     index=net.trafo.index
#     )
    
#     Z_ref_trafo = net.trafo['vn_lv_kv']**2 * net.sn_mva / net.trafo['sn_mva']

#     # 1. series losses:joule or heat losses
#     # short-circuit resistance r
#     net.trafo['r_per_unit'] = (net.trafo['vkr_percent']/100 * net.sn_mva / net.trafo['sn_mva']) * Z_ref_trafo / Z_N

#     # short-circuit impedance 
#     zmag_per_unit = (net.trafo['vk_percent']/100 * net.sn_mva / net.trafo['sn_mva']) * Z_ref_trafo / Z_N

#     # short-circuit impedance
#     net.trafo['x_per_unit'] = np.sqrt(zmag_per_unit**2 - net.trafo['r_per_unit']**2)    
#     if is_rq2: 
#         net.trafo['X_ohm'] = net.trafo['x_per_unit'] * Z_N / Z_ref_trafo

#     # 2. magnetising losses: eddy currents and hysterisis 
#     # magnetising conductance g
#     net.trafo['g_per_unit'] = (net.trafo['pfe_kw']/(net.trafo['sn_mva'] * 1000) * net.sn_mva / net.trafo['sn_mva']) * Z_N / Z_ref_trafo

#     # shunt admittance 
#     y_sh = (net.trafo['i0_percent']/100) * (net.sn_mva / net.trafo['sn_mva']) * Z_N / Z_ref_trafo


#     # magnetising impedance b
#     net.trafo['b_per_unit'] = np.sqrt(y_sh**2- net.trafo['g_per_unit']**2)

#     # add tap ratios 
#     if any(net.trafo.tap_phase_shifter):
#         print(f"Network has {sum(net.trafo.tap_phase_shifter)} phase-shifting transformers.")

#     if "tap_pos" in net.trafo.columns: # does not exist 
#         print("No column named trafo.tap_pos, using tap_ratios...")
#         V_ref_lv_trafo = net.trafo['vn_lv_kv'].values 
#         V_ref_hv_trafo = net.trafo['vn_hv_kv'].values

#         V_ref_lv_bus = net.bus['vn_kv'].loc[net.trafo['lv_bus'].values].values
#         V_ref_hv_bus = net.bus['vn_kv'].loc[net.trafo['hv_bus'].values].values

#         net.trafo['tap_ratio'] = (V_ref_hv_trafo * V_ref_lv_bus) / (V_ref_lv_trafo * V_ref_hv_bus)
#         net.trafo['n_tap'] = round(net.trafo['tap_ratio'])
#     else:
#         if sum(net.trafo.tap_pos.isna()) > 0: 
#             print("Filling nan as 0 in tap_pos")
#             net.trafo.tap_pos.fill_na(0, inplace=True)
#         # pandapower formula
#         net.trafo['n_tap'] = 1 + (net.trafo.tap_pos - net.trafo.tap_neutral) * (net.trafo.tap_step_percent / 100)    
#     return net



# def get_branch_admittance(net: pp.pandapowerNet) -> Tuple: 
#     """
#     Calculates the complex series admittance for each branch as an effective line. 
#     # TODO: Tap-changers. 
#     """

#     ################ LINE PARAMETERS ########################   
#     sys_freq = 50 # Hz 

#     # Per-Unit values 
#     # convert r_ohm_per_km to r_per_unit
#     r_pu = (net.line['r_ohm_per_km'] * net.line['length_km'] / net.line['parallel']) / (net.bus['vn_kv'].loc[net.line['from_bus'].values]**2/net.sn_mva).values 

#     # convert x_ohm_per_km to x_ohm
#     x_pu = (net.line['x_ohm_per_km'] * net.line['length_km'] / net.line['parallel']) / (net.bus['vn_kv'].loc[net.line['from_bus'].values]**2/net.sn_mva).values

#     # convert c_nf_per_km to b_pu
#     b_pu = ( 2 * np.pi * sys_freq * net.line['c_nf_per_km'] * 10**(-9) * net.line['length_km'] * net.line['parallel']) * (net.bus['vn_kv'].loc[net.line['from_bus'].values]**2/net.sn_mva).values

#     # convert g_us_per_km to g_pu
#     g_pu = ( 2 * np.pi * sys_freq * net.line['g_us_per_km'] * 10**(-6) * net.line['length_km'] * net.line['parallel']) * (net.bus['vn_kv'].loc[net.line['from_bus'].values]**2/net.sn_mva).values 


#     z_pu = r_pu + 1j*x_pu
#     y_series_pu = 1 / z_pu # series admittance 


#     y_shunt_pu = g_pu - 1j*b_pu # shunt admittance 

#     a_1_pu = y_series_pu + y_shunt_pu/2
#     a_2_pu = - y_series_pu
#     a_3_pu = - y_series_pu
#     a_4_pu = y_series_pu + y_shunt_pu/2

#     for idx, line in enumerate(line_tuple): 
#         y_series[line] = y_series_pu[idx]
#         y_shunt[line] = y_shunt_pu[idx]
    
#     yb_size = len(net.bus)

#     Ybline = np.zeros((yb_size, yb_size)).astype(complex)

#     fb_line = net.line.from_bus 
#     tb_line = net.line.to_bus 

#     line_idx = net.line.index 

#     for (idx, fb, tb) in zip(line_idx, fb_line, tb_line): 
#         Ybline[fb, fb] = complex(a_1_pu[idx])
#         Ybline[fb, tb] = complex(a_2_pu[idx])
#         Ybline[tb, fb] = complex(a_3_pu[idx])
#         Ybline[tb, tb] = complex(a_4_pu[idx])

#     # get the pandpaower internal YBus 
#     pp.runpp(net)
#     Ybus = np.array(net._ppc["internal"]["Ybus"].todense())

#     ################# TRANSFORMER PARAMETERS ######################
#     # derived from Ybus matrix. 
#     Ybus_trafo = Ybus - Ybline

#     if sum(net.trafo.tap_pos.isna()) > 0: 
#         print("Filling nan as 0 in tap_pos, tap_neutral, tap_step_degree")
#         net.trafo.loc[net.trafo.loc[:,'tap_pos'].isna(),'tap_pos'] = 0
#         net.trafo.loc[net.trafo.loc[:,'tap_neutral'].isna(),'tap_neutral'] = 0
#         net.trafo.loc[net.trafo.loc[:,'tap_step_degree'].isna(),'tap_step_degree'] = 0
        
#     ps_s = net.trafo.shift_degree * np.pi/180
#     tap_nom_s = 1 + (net.trafo.tap_pos - net.trafo.tap_neutral) * (net.trafo.tap_step_percent / 100)
#     N_tap_s = tap_nom_s * np.exp(1j*ps_s)

#     for id, trafo_edge in zip(net.trafo.index.values, trafo_tuple):
#         hv_bus, lv_bus = trafo_edge 
#         a_1t_pu = Ybus_trafo[hv_bus, hv_bus]
#         a_2t_pu = Ybus_trafo[hv_bus, lv_bus]
#         a_3t_pu = Ybus_trafo[lv_bus, hv_bus]
#         a_4t_pu = Ybus_trafo[lv_bus, lv_bus]

#         # series impedance 
#         y_series[trafo_edge] = - a_3t_pu * N_tap_s[id]

#         # shunt impedance 
#         y_shunt[trafo_edge] = 2* (a_4t_pu - y_series[trafo_edge]) 


#     return y_series, y_shunt