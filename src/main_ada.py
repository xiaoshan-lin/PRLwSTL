import argparse
import create_environment_new as ce # UPDATE
import timeit, time
import networkx as nx
import matplotlib.pyplot as plt
import yaml
import os,sys
import datetime
from tkinter.messagebox import askyesno
import shutil

from pyTWTL.twtl_to_dfa_test import twtl_to_dfa
from pyTWTL import lomap
from pyTWTL import synthesis_test as synth

import copy
import numpy as np
import random
from tmdp_stl import Tmdp
from product_automaton_ada import AugPa
from fmdp_stl import Fmdp
from static_reward_mdp import StaticRewardMdp
from STL import STL
import Q_learning as ql
import Q_learning_ada as ql_ada

this_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_file_path+'/../plot')
from plot_result import plot_result
from plot_env import plot_env

def build_environment(env_cfg, twtl_cfg, mdp_type, reward_cfg, verbose, print_info, lower_bound, incr):
    """
    Create the MDP, Augmented MDP, DFA, and Augmented Product MDP 

    Parameters
    ----------
    env_cfg : dict
        Environment configuration dictionary
    twtl_cfg : dict
        TWTL constraint configuration dictionary
    mdp_type : string
        The MDP augmentation type. Either 'static rewards', 'flag-MDP', or 'tau-MDP'
    reward_cfg : dict
        Reward configuration dictionary. Expected items depends on mdp_type
    
    Returns
    -------
    AugPa
        An Augmented Product MDP
    """

    # Get values from configs
    m = env_cfg['height']
    n = env_cfg['width']
    h = 1   # depth: use only 2 dimensions
    init_state = env_cfg['init state']
    obstacles = env_cfg['obstacles']

    custom_task = twtl_cfg['custom task']
    if custom_task == 'None':
        custom_task = None

    # UPDATE
    custom_prop = env_cfg['custom prop']
    if custom_prop == True:
        custom_prop_dict = env_cfg['custom prop dict']
    else:
        custom_prop_dict = None

    def xy_to_region(x,y,z):
        # x is down, y is across
        # ignore 3rd dim
        return x * n + y

    # TODO: Clean this up

    # m: length, n: width

    # =================================
    # MDP Creation
    # =================================
    ts_start_time = timeit.default_timer()
    disc = 1
    TS, obs_mat, state_mat = ce.create_ts(m,n,h)	
    path = '../data/ts_' + str(m) + 'x' + str(n) + 'x' + str(h) + '_1Ag_1.txt'
    abs_path = os.path.join(this_file_path, path)
    paths = [abs_path]
    # bases = {init_state: 'Base1'}
    bases = {}
    obs_mat = ce.update_obs_mat(obs_mat, state_mat, m, obstacles, init_state)
    TS      = ce.update_adj_mat_3D(m, n, h, TS, obs_mat)
    ce.create_input_file(TS, state_mat, obs_mat, custom_prop_dict, paths[0], bases, disc, m, n, h, 0)
    ts_file = paths
    ts_dict = lomap.Ts(directed=True, multi=False) 
    ts_dict.read_from_file(ts_file[0])
    ts = synth.expand_duration_ts(ts_dict)
    init_state_str = 'r' + str(xy_to_region(*init_state))
    ts.init = {init_state_str: 1}

    # =================================
    # Signal Creation
    # =================================
    # create dictionary mapping mdp states to a position signal for use in robustness calculation
    # apply offset so pos is middle of region
    state_to_pos = dict()
    dims = ['x', 'y', 'z']
    for s in ts.g.nodes():
        num = int(s[1:])
        #TODO: 3rd dim?
        pos = ((num // n) + 0.5, (num % n) + 0.5, 0.5)
        state_to_pos[s] = {d:p for d,p in zip(dims, pos)}
    # stl.set_ts_sig_dict(state_to_pos)
    
    ts_timecost =  timeit.default_timer() - ts_start_time


    # =================================
    # DFA Creation
    # =================================
    dfa_start_time = timeit.default_timer()
    region_coords = twtl_cfg['regions']
    if custom_task not in ['None', None]:
        # Build the spec from regions in config

        for r,c in region_coords.items():
            region_num = xy_to_region(*c)
            custom_task = custom_task.replace(r, 'r' + str(region_num))
        phi = custom_task
    else:
        pickup = region_coords['pickup']
        delivery = region_coords['delivery']

        pick_up_reg = xy_to_region(*pickup)
        delivery_reg = xy_to_region(*delivery)
        pick_up_str  = str(pick_up_reg)
        delivery_str = str(delivery_reg)

        twtl_horizon = twtl_cfg['time horizon']
        tf1 = int((twtl_horizon-1)/2) # time bound
        tf2 = int(twtl_horizon) - tf1 - 1
        phi = '[H^1 r' + pick_up_str + ']^[0, ' +  str(tf1) + '] * [H^1 r' + delivery_str + ']^[0,' + str(tf2) + ']'
    out = twtl_to_dfa(phi, kind='infinity', norm=True)
    dfa_inf = out['infinity']
    dfa_timecost =  timeit.default_timer() - dfa_start_time
    bounds = out['bounds']
    dfa_horizon = bounds[-1]
    if custom_task == None and dfa_horizon != twtl_horizon:
        raise RuntimeError(f'Received unexpected time bound from DFA. DFA horizon: {dfa_horizon}, expected: {twtl_horizon}.')
    else:
        twtl_horizon = dfa_horizon

    # add self edge to accepting state
    # All observation cases in accepting state should result in self edge
    input_set = dfa_inf.alphabet    
    for s in dfa_inf.final:
        dfa_inf.g.add_edge(s,s, guard='(else)', input=input_set, label='(else)', weight=0)
    
    # print(phi)
    # A = nx.nx_agraph.to_agraph(dfa_inf.g)
    # A.layout(prog='dot')
    # A.draw('dfa.png')

    # plt.subplot()
    # nx.draw(dfa_inf.g, with_labels=True)
    # plt.show()

    # exit()

    # =================================
    # Augmented MDP Creation
    # =================================
    aug_mdp_timer = timeit.default_timer()
    if mdp_type == 'flag-MDP':
        stl_expr = reward_cfg['STL expression']
        aug_mdp = Fmdp(ts, stl_expr, state_to_pos)
    elif mdp_type == 'tau-MDP':
        stl_expr = reward_cfg['STL expression']
        aug_mdp = Tmdp(ts, stl_expr, state_to_pos)
    elif mdp_type == 'static rewards':
        hrz = dfa_horizon
        reward_dict = reward_cfg['reward dict']
        aug_mdp = StaticRewardMdp(ts, hrz, state_to_pos, reward_dict)
    else:
        raise ValueError("invalid AUG_MDP_TYPE")
    aug_mdp_timecost = timeit.default_timer() - aug_mdp_timer

    mdp_horizon = aug_mdp.get_hrz()
    
    if mdp_horizon != twtl_horizon:
        if mdp_type == 'static rewards':
            raise RuntimeError(f'Static rewards MDP has an incorrect time horizon. This is likely \
                an implementation error. MDP horizon: {mdp_horizon}, TWTL horizon: {twtl_horizon}')
        else:
            raise ValueError(f'STL and TWTL time horizon mismatch. Please adjust either spec \
                so the horizons match. STL time horizon: {mdp_horizon}, TWTL time horizon: {twtl_horizon}')


    # =================================
    # Augmented Product MDP Creation
    # =================================
    pa_start_time = timeit.default_timer()
    pa_or = AugPa(aug_mdp, dfa_inf, twtl_horizon, lower_bound, incr)
    
    pa = copy.deepcopy(pa_or)	      # copy the pa
    pa_timecost =  timeit.default_timer() - pa_start_time

    # Compute the energy of the states
    energy_time = timeit.default_timer()
    pa.compute_energy()
    energy_timecost =  timeit.default_timer() - energy_time

    init_state_num = init_state[0] * n + init_state[1]

    # =================================
    # Print information
    # =================================
    if print_info:
        print('##### PICK-UP and DELIVERY MISSION #####' + "\n")
        print('Initial Location  : ' + str(init_state) + ' <---> Region ' + str(init_state_num))
        if custom_task == None:
            print('Pick-up Location  : ' + str(pickup) + ' <---> Region ' + pick_up_str)
            print('Delivery Location : ' + str(delivery) + ' <---> Region ' + delivery_str)
        else:
            for r,c in region_coords.items():
                rnum = xy_to_region(*c)
                print(f'{r} : {c} <---> Region {rnum}')
        # print('Reward Locations  : ' + str(rewards) + ' <---> Regions ' + str(rewards_ts_indexes) + "\n")
        print('State Matrix : ')
        print(state_mat)
        print("\n")
        print('Mission Duration  : ' + str(twtl_horizon) + ' time steps')
        print('TWTL Task : ' + phi + "\n")
    
    if verbose:
        print('Time PA state size: {}\n'.format(pa.get_tpa_state_size()))
        print('Time Cost:')
        print('TS creation time (s):            {:<7}'.format(ts_timecost))
        print('Augmented MDP creation time (s): {:<7}'.format(aug_mdp_timecost))
        # print('			TS created in ' + str(ts_timecost) + ' seconds')
        print('DFA creation time (s):           {:<7}'.format(dfa_timecost))
        print('PA creation time (s):            {:<7}'.format(pa_timecost))
        print('PA energy calculation time (s):  {:<7}'.format(energy_timecost))

    return pa


def main(verbose=True, function='l'):
    """
    Main function. Read in configuration values, construct the Pruned Time-Product MDP, 
    find the optimal policy, and test the optimal policy.
    """
    try: 
        if function=='p':
             plot_flag, proj_dir = plot_result('')
             if not plot_flag:
                 print('Result plotting is canceled')

             if not proj_dir:
                 plot_flag, proj_dir = plot_env('')
             else:
                 plot_flag, proj_dir = plot_env(proj_dir)

             if not plot_flag:
                 print('Env plotting is canceled')
                 sys.exit() 
                
        elif function=='l':
            my_path = os.path.dirname(os.path.abspath(__file__))
            def_cfg_rel_path = '../configs/default.yaml'
            # Load default config
            def_cfg_rel_path = '../configs/new.yaml'
            def_cfg_path = os.path.join(my_path, def_cfg_rel_path)
            with open(def_cfg_path, 'r') as f:
                config = yaml.safe_load(f)

            # stl_expr = 'G[0,10]F[0,3](((x>1)&(x<2))&((y>3)&(y<4)))'

            # ==== Read in configuration values ====
            # Q-learning config
            qlearn_cfg = config['Q-learning config']
            num_episodes = qlearn_cfg['number of episodes']     # of episodes
            learn_rate = qlearn_cfg['learning rate']
            discount   = qlearn_cfg['discount']

            explore_prob_start = qlearn_cfg['explore probability start']
            explore_prob_end = qlearn_cfg['explore probability end']
            # start * decay^(num_eps - 1) = end
            explore_prob_decay = (explore_prob_end/explore_prob_start)**(1/(num_episodes-1))

            # environment config
            env_cfg = config['environment']
            eps_unc    = env_cfg['real action uncertainty'] # Uncertainity in actions, real uncertainnity in MDP
            eps_unc_learning = env_cfg['over estimated action uncertainty'] # Overestimated uncertainity used in learning

            # TWTL constraint config
            twtl_cfg = config['TWTL constraint']
            des_prob = twtl_cfg['desired satisfaction probability'] # Minimum desired probability of satisfaction

            mdp_type = config['MDP type']
            if mdp_type == 'static rewards':
                reward_cfg = config['static rewards']
            else:
                reward_cfg = config['aug-MDP rewards']

            ada_cfg = config['Ada']
            ada = ada_cfg['ada']
            switching = ada_cfg['switching']
            ada_alpha = ada_cfg['ada_alpha']
            save_data = ada_cfg['save_data']
            safe_pad = ada_cfg['safe_pad']
            n_samples = ada_cfg['n_samples']
            stopping = ada_cfg['stopping']
            repeat = ada_cfg['repeat']
            lower_bound = ada_cfg['lower_bound']
            incr = ada_cfg['incr']
            double = ada_cfg['double']
       
            test_cfg = config['Testing']
            test_iters = test_cfg['test_iters']
            test_result = {i:{} for i in range(repeat)}
            assert ada*switching != 1, "ada and switching can not be both True"

            if switching:
                proj_dir = os.path.join(this_file_path,'../result/switching')
            elif ada:
                proj_dir = os.path.join(this_file_path,'../result/ada')
            else:
                proj_dir = os.path.join(this_file_path,'../result/constant')
            dt = datetime.datetime.today()
            proj_dir = os.path.join(proj_dir, '{}_{}_{}_{}_{}'.format(dt.month, dt.day,dt.hour,dt.minute,dt.second))
            proj_dir = os.path.abspath(proj_dir)
            os.mkdir(proj_dir)
            shutil.copy(def_cfg_path, os.path.join(proj_dir,'default.yaml'))

            # test_gen_time()

            # ==== Construct the Pruned Time-Product MDP ====
            print_info = True
            for idx in range(repeat):
                
                prep_start_time = timeit.default_timer()
                # Construct the Product MDP
                pa = build_environment(env_cfg, twtl_cfg, mdp_type, reward_cfg, verbose, print_info, lower_bound, incr)
                
                pa.get_env_size(env_cfg['height'], env_cfg['width'])
                
                # Prune it at each time step
                prune_start = timeit.default_timer()
                print('\nLearning: {} round'.format(idx+1))
                if not switching:
                    if not ada:
                        pa.prune_actions(eps_unc_learning, des_prob)
                        prune_end = gen_start = timeit.default_timer()
                        if verbose:
                            print('Time PA action pruning time (s): {}'.format(prune_end - prune_start))
                    else:
                        pa.reset_actions()
                        prune_end = gen_start = timeit.default_timer()
                        if verbose:
                            print('Time PA setting time (s): {}'.format(prune_end - prune_start))
                else:
                    pa.reset_actions() 
                    print('Prune PA actions:')
                    pa.prune_actions(eps_unc_learning, des_prob)
                    prune_end = gen_start = timeit.default_timer()
                    if verbose:
                        print('Time PA action pruning time (s): {}'.format(prune_end - prune_start))
                pa.gen_new_ep_states()
                gen_end = timeit.default_timer()
                prep_end_time = timeit.default_timer()
                
                if verbose:
                    print('New ep/traj generation time (s): {}'.format(gen_end - gen_start))
                    print('')
                print('Total environment creation time: {}'.format(prep_end_time - prep_start_time))
                print('')

                # ==== Find the optimal policy ====
                print('learning with {} episodes'.format(num_episodes))
                timer = timeit.default_timer()
                
                if not double:
                    pi = ql_ada.Q_learning_ada(pa, num_episodes, eps_unc, learn_rate, discount, explore_prob_decay,  
                                  explore_prob_start,eps_unc_learning, des_prob, proj_dir, ada, switching, ada_alpha, save_data, 
                                  safe_pad,n_samples,stopping, idx)
                else:
                    pi_1, pi_2, b_ub_ratio = ql_ada.Q_learning_double(pa, num_episodes, eps_unc, learn_rate, discount,
                                  explore_prob_decay, explore_prob_start,eps_unc_learning, des_prob, proj_dir, ada, switching, 
                                  ada_alpha, save_data, safe_pad,n_samples,stopping, idx) 
                    print(pi_1)
                    print(pi_2)
                    print(b_ub_ratio)
                qlearning_time = timeit.default_timer() - timer
                print('learning time: {} seconds'.format(qlearning_time))

                # ==== test policy ====
                stl_expr = config['aug-MDP rewards']['STL expression']
                if not double:
                    _,twtl_sat_rate, avg_reward = ql_ada.test_policy(pi, pa, stl_expr, eps_unc, test_iters, mdp_type, idx, proj_dir)
                else:
                    _,twtl_sat_rate, avg_reward = ql_ada.test_policy_double(pi_1, pi_2, b_ub_ratio, pa, stl_expr, eps_unc, test_iters,
                                                     mdp_type, idx, proj_dir)
                    
                test_result[idx]['suc_rate']=twtl_sat_rate
                test_result[idx]['reward']=avg_reward

                print_info = False

            if config['Visualize']['result']:
                    plot_result(proj_dir) 
            answer = askyesno(title='Confirmation',
                              message='Do you want to save the data?')
            if not answer:
                shutil.rmtree(proj_dir)
            else:
                with open(os.path.join(proj_dir,'test_result.yaml'), 'w') as yaml_file:
                    yaml.dump(test_result,yaml_file)
        
    except KeyboardInterrupt:
        answer = askyesno(title='Confirmation',
                          message='Do you want to save the data?')
        if not answer:
            print('removing project directory ...')
            shutil.rmtree(proj_dir)
        sys.exit()

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description ='Specifying desired action, currently support two functions:\n\
                                                   1. Learn a new task; 2. Plot saved data',
                                     usage='--mode [l,p]')
    parser.add_argument('--mode',required={'l','p'}, help ='l: Learn a new task; p: Plot saved data')
    args = parser.parse_args()
    main(verbose=True, function=args.mode)
