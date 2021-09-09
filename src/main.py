#from create_environment import create_ts, update_obs_mat, update_adj_mat_3D,\
#								create_input_file, update_adj_mat_3D

from __future__ import division # dividing two integers produces a float

import create_environment as ce
import timeit, time
import lomap
import twtl
import synthesis as synth
import networkx as nx
from dfa import DFAType
import copy
import numpy as np
import random
from tmdp_stl import Tmdp
from product_automaton import AugPa
from fmdp_stl import Fmdp



AUG_MDP_TYPE = 'FLAG-MDP'
# AUG_MDP_TYPE = 'TAU-MDP'

NORMAL_COLOR = '\033[0m'

COLOR_DICT = {
    'pi epsilon go' : '\033[38;5;3m',   # yellow
    'explore'       : '\033[38;5;4m',   # blue
    'exploit'       : '\033[38;5;2m',   # green
    'intended'      : '\033[49m',       # no highlight
    'unintended'    : '\033[48;5;1m'    # red highlight
}




def tau_mdp(ts, ts_weighted, tau):

    # def next_mdp_states(state):
    # 	return ts.g.edge[state].keys()
        # all_states = adj_mat[state]
        # adj_states = []
        # for state,cost in enumerate(all_states):
        # 	if cost > 0:
        # 		adj_states.append(state)
        # return adj_states

    def build_states(past, ts_edge_dict, tau):
        if tau == 1:
            # One tau-MDP state per MDP state
            return [tuple(past)]
        next_states = ts_edge_dict[past[-1]]
        if len(next_states) == 0:
            # no next states. Maybe an obstacle?
            return []
        tmdp_states = [past + [ns] for ns in next_states]
        if len(tmdp_states[0]) == tau:
            # each tau-MDP state has 'tau' MDP states
            # make each state unmutable
            return [tuple(s) for s in tmdp_states]
        
        # recurse for each state in states
        more_tmdp_states = []
        for x in tmdp_states:
            more_tmdp_states.extend(build_states(x, ts_edge_dict, tau))
        
        return more_tmdp_states
    
    # Make a dictionary of ts edges and add state for null history that can transition to any state
    ts.g.add_edges_from([(None,s,{'weight':1}) for s in ts.g.edge.keys()])
    ts.g.add_edge(None,None, weight=0)
    ts_weighted.g.add_edges_from([(None,s,{'duration':1,'edge_weight':0}) for s in ts.g.edge.keys()])
    ts_weighted.g.add_edge(None,None, duration=1, edge_weight=0)
    ts_edge_dict = {s:ts.g.edge[s].keys() for s in ts.g.edge.keys()}
    # ts_edge_dict[None] = ts.g.edge.keys() + [None]

    # make list of tau mdp states where each state is represented by a tuple of mdp states
    tmdp_states = []
    for s in ts_edge_dict.keys():
        tmdp_states.extend(build_states([s], ts_edge_dict, tau))

    # tmdp_states.remove((None,) * tau) # No state should end with a null

    # try and recreate process used in ts.read_from_file() except with tau mdp
    # There seems to be a ts with only weights 1 and ts_dict with original weights
    # Looks like it will be easiest to create another nx for weights rather than recreate desired output format
    tmdp = lomap.Ts(directed=True, multi=False)
    tmdp_weighted = lomap.Ts(directed=True, multi=False)
    tmdp.name = tmdp_weighted.name = "Tau MDP"
    tmdp.init = tmdp_weighted.init = {((None,) * (tau-1)) + (ts.init.keys()[0],) :1}

    # create dict of dicts representing edges and attributes of each edge to construct the nx graph from
    # attributes are based on the mdp edge between the last (current) states in the tau mdp sequence
    edge_dict = {}
    edge_dict_weighted = {}
    for x1 in tmdp_states:
        # if x1[-1] == None:
        # 	# tmdp none state should not show up in usage. It is necessary for pa construction.
        # 	edge_attrs = {s:{None:None} for s in ts.g.nodes()}
        # 	edge_attrs[None] = {None:None}
        # 	edge_attrs_weighted = {s:{0:{None:None}} for s in edge_attrs.keys()}
        # else:
        edge_attrs = ts.g.edge[x1[-1]]
        edge_attrs_weighted = ts_weighted.g.edge[x1[-1]]
        # tmdp states are adjacent if they share the same (offset) history. "current" state transition is implied valid 
        # based on the set of names created
        if tau > 1:
            #TODO use next_mdp_states instead of conditional
            edge_dict[x1] = {x2:edge_attrs[x2[-1]] for x2 in tmdp_states if x1[1:] == x2[:-1]}
            edge_dict_weighted[x1] = {x2:edge_attrs_weighted[x2[-1]][0] for x2 in tmdp_states if x1[1:] == x2[:-1]}
        else:
            # Case of tau = 1
            edge_dict[x1] = {(x2,):edge_attrs[x2] for x2 in ts_edge_dict[x1[0]]}
            edge_dict_weighted[x1] = {(x2,):edge_attrs_weighted[x2][0] for x2 in ts_edge_dict[x1[0]]}

    tmdp.g = nx.from_dict_of_dicts(edge_dict, create_using=nx.MultiDiGraph()) 
    tmdp_weighted.g = nx.from_dict_of_dicts(edge_dict_weighted, create_using=nx.MultiDiGraph()) 

    # add node attributes based on last state in sequence
    for n in tmdp.g.nodes():
        tmdp.g.node[n] = ts.g.node[n[-1]]
    for n in tmdp_weighted.g.nodes():
        tmdp_weighted.g.node[n] = ts_weighted.g.node[n[-1]]

    return(tmdp, tmdp_weighted)
    


def build_environment(m, n, h, init_state, pick_up, delivery, custom_task, stl_expr):

    def xy_to_region(x,y,z):
        # x is down, y is across
        # ignore 3rd dim
        return x * n + y

    # TODO: Clean this up

    # m: length, n: width

    # Create the environment and get the TS #
    ts_start_time = timeit.default_timer()
    disc = 1
    TS, obs_mat, state_mat = ce.create_ts(m,n,h)	
    path = '../data/ts_' + str(m) + 'x' + str(n) + 'x' + str(h) + '_1Ag_1.txt'
    paths = [path]
    # bases = {init_state: 'Base1'}
    bases = {}
    obstacles = []
    obs_mat = ce.update_obs_mat(obs_mat, state_mat, m, obstacles, init_state)
    TS      = ce.update_adj_mat_3D(m, n, h, TS, obs_mat)
    ce.create_input_file(TS, state_mat, obs_mat, paths[0], bases, disc, m, n, h, 0)
    ts_file = paths
    ts_dict = lomap.Ts(directed=True, multi=False) 
    ts_dict.read_from_file(ts_file[0])
    ts = synth.expand_duration_ts(ts_dict)
    init_state_str = 'r' + str(xy_to_region(*init_state))
    ts.init = {init_state_str: 1}

    # create dictionary mapping mdp states to a position signal for use in satisfaction calculation
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

    # Create a tau MDP by combining "past" TS states
    # Variables to recreate
    # nx.get_edge_attributes(ts_dict.g,'edge_weight')
    # ts
    # There seems to be a ts with only weights 1 and ts_dict with original weights
    # Looks like it will be easiest to create another nx for weights rather than recreate desired output format

    # tmdp, _ = tau_mdp(ts, ts_dict, tau)

    if AUG_MDP_TYPE == 'FLAG-MDP':
        aug_mdp = Fmdp(ts, stl_expr, state_to_pos)
    elif AUG_MDP_TYPE == 'TAU-MDP':
        aug_mdp = Tmdp(ts, stl_expr, state_to_pos)
    else:
        raise Exception("invalid AUG_MDP_TYPE")

    # TODO: check that this is the correct episode length
    ep_len = aug_mdp.get_hrz()


    # Get the DFA #
    dfa_start_time = timeit.default_timer()
    pick_up_reg = xy_to_region(*pick_up)
    delivery_reg = xy_to_region(*delivery)
    pick_up_str  = str(pick_up_reg)
    delivery_str = str(delivery_reg)
    if custom_task != None:
        phi = custom_task
        raise Exception("Error: Custom task time bound compatability checking is not yet implemented!")
    else:
        tf1 = int((ep_len-1)/2) # time bound
        tf2 = int(ep_len) - tf1 - 1
        phi = '[H^1 r' + pick_up_str + ']^[0, ' +  str(tf1) + '] * [H^1 r' + delivery_str + ']^[0,' + str(tf2) + ']'  # Construc the task according to pickup/delivery )^[0, ' + tf + ']'
    _, dfa_inf, _ = twtl.translate(phi, kind=DFAType.Infinity, norm=True) # states and sim. time ex. phi = '([H^1 r47]^[0, 30] * [H^1 r31]^[0, 30])^[0, 30]' 
    dfa_timecost =  timeit.default_timer() - dfa_start_time # DFAType.Normal for normal, DFAType.Infinity for relaxed

    # add self edge to accepting state
    # TODO probably don't hardcode the input set. see dfa_inf.alphabet
    dfa_inf.g.add_edge(4,4, {'guard': '(else)', 'input':set([0,1,2,3]), 'label':'(else)', 'weight':0})

    # Get the PA #
    pa_start_time = timeit.default_timer()
    # alpha = 1       # with alpha = 1, all transitions are weighted equally in energy calculation
    # nom_weight_dict = {}
    # weight_dict = {}
    # pa_or = ts_times_fsa(ts, dfa_inf) # Original pa
    # real_init = tmdp.init
    # pa_or = synth.ts_times_fsa(tmdp, dfa_inf) # Original pa
    pa_or = AugPa(aug_mdp, dfa_inf, ep_len)
    # tmdp.init = real_init
    # pa_or.g.remove_node(((None,) * tau, 0))

    # TODO: incorporate this into Pa class
    # edges_all = nx.get_edge_attributes(ts_dict.g,'edge_weight')
    # edges_all = nx.get_edge_attributes(tmdp_weighted.g,'edge_weight')
    # max_edge = max(edges_all, key=edges_all.get)
    # norm_factor = edges_all[max_edge]
    # for pa_edge in pa_or.g.edges():
    #     mdp_s_1 = pa_or.get_mdp_state(pa_edge[0])
    #     mdp_s_2 = pa_or.get_mdp_state(pa_edge[1])
    #     edge = (mdp_s_1, mdp_s_2, 0)
    #     nom_weight_dict[pa_edge] = edges_all[edge]/norm_factor
    # nx.set_edge_attributes(pa_or.g, 'edge_weight', nom_weight_dict)
    # nx.set_edge_attributes(pa_or.g, 'weight', 1)
    pa = copy.deepcopy(pa_or)	      # copy the pa
    # time_weight = nx.get_edge_attributes(pa.g,'weight')
    # edge_weight = nx.get_edge_attributes(pa.g,'edge_weight')
    # for pa_edge in pa.g.edges():
    #     weight_dict[pa_edge] = alpha*time_weight[pa_edge] + (1-alpha)*edge_weight[pa_edge]
    # nx.set_edge_attributes(pa.g, 'new_weight', weight_dict)     # new_weight is used in energy computation
    pa_timecost =  timeit.default_timer() - pa_start_time

    # Compute the energy of the states #
    energy_time = timeit.default_timer()
    pa.compute_energy()

    energy_timecost =  timeit.default_timer() - energy_time

    init_state_num = [init_state[0] * n + init_state[1]]

    # # Display some important info
    print('##### PICK-UP and DELIVERY MISSION #####' + "\n")
    print('Initial Location  : ' + str(init_state) + ' <---> Region ' + str(init_state_num))
    print('Pick-up Location  : ' + str(pick_up) + ' <---> Region ' + pick_up_str)
    print('Delivery Location : ' + str(delivery_state[0]) + ' <---> Regions ' + delivery_str)
    # print('Reward Locations  : ' + str(rewards) + ' <---> Regions ' + str(rewards_ts_indexes) + "\n")
    print('State Matrix : ')
    print(state_mat)
    print("\n")
    print('Mission Duration  : ' + str(ep_len) + ' time steps')
    print('TWTL Task : ' + phi + "\n")
    print('Computational Costst : TS created in ' + str(ts_timecost) + ' seconds')
    # print('			TS created in ' + str(ts_timecost) + ' seconds')
    print('		       DFA created in ' + str(dfa_timecost) + ' seconds')
    print('		       PA created in ' + str(pa_timecost) + ' seconds')
    print('		       Energy of PA states calculated in ' + str(energy_timecost) + ' seconds')

    return pa

def Q_learning(pa, episodes, eps_unc, learn_rate, discount, eps_decay, epsilon, samples):

    # Log state sequence and reward
    trajectory_reward_log = []
    mdp_traj_log = ''
    tr_log_file = '../output/trajectory_reward_log.txt'
    mdp_log_file = '../output/mdp_trajectory_log.txt'
    # q_table_file = '../output/live_q_table.txt'
    log = True
    # truncate file
    open(tr_log_file, 'w').close()
    open(mdp_log_file, 'w').close()

    # Count trajectories that reach an accepting state (pass the TWTL task)
    twtl_pass_count = 0

    # initialize Q table
    init_val = -10
    time_steps = pa.get_hrz()
    qtable = [{} for _ in range(time_steps+1)]
    for i,_ in enumerate(qtable):
        qtable[i] = {p:{} for p in pa.get_states()}
        for p in qtable[i]:
            qtable[i][p] = {q:init_val + np.random.normal(0,0.0001) for q in pa.g.neighbors(p)}

    # initialize optimal policy pi on pruned time product automaton
    pi = [{} for _ in range(time_steps)]
    for t,_ in enumerate(pi):
        for p in pa.pruned_time_actions[t]:
            if pa.pruned_time_actions[t][p] != []:
                # initialize with a neighbor in the pruned space
                pi[t][p] = max(pa.pruned_time_actions[t][p], key=qtable[t][p].get)
            else:
                # Empty action set. No actions available in the pruned time pa.
                pi[t][p] = None
        # pi[t] = {p:max(qtable[t][p], key=qtable[t][p].get) for p in pa.pruned_time_actions[t]}

    # initial state,time
    z,t_init,init_traj = pa.initial_state_and_time()
    if log:
        trajectory_reward_log.extend(init_traj)
        init_mdp_traj = [pa.get_mdp_state(z) for z in init_traj]
        for x in init_mdp_traj:
            mdp_traj_log += '{:<4}'.format(x)
    # z = pa.init.keys()[0]

    # Loop for number of training episodes
    for _ in range(episodes):
        for t in range(t_init, time_steps):

            next_states = pa.pruned_time_actions[t][z]
            if next_states == []:
                # Pruned action set is empty, choose action with lowest energy
                probable_z = pa.pi_eps_go[t][z]
                action_chosen_by = "pi epsilon go"
            else:
                if np.random.uniform() < epsilon:   # Explore
                    probable_z = random.choice(next_states)
                    action_chosen_by = "explore"
                else:                               # Exploit
                    probable_z = pi[t][z]
                    action_chosen_by = "exploit"
            
            # Take the action, result may depend on uncertainty
            next_z = pa.take_action(z, probable_z, eps_unc)
            if next_z == probable_z:
                action_result = 'intended'
            else:
                action_result = 'unintended'

            reward = pa.reward(next_z)
            cur_q = qtable[t][z][next_z]
            # TODO on the last time step, next z should probably be the reset version
            future_qs = qtable[(t+1) % (time_steps + 1)][next_z] # TODO: evaluate if a modulus is the correct approach
            max_future_q = max(future_qs.values())

            # Update q value
            new_q = (1 - learn_rate) * cur_q + learn_rate * (reward + discount * max_future_q)
            qtable[t][z][next_z] = new_q

            # Update optimal policy
            if next_states != []:
                pi[t][z] = max(next_states, key=qtable[t][z].get)

            if log:
                trajectory_reward_log.append(next_z)
                mdp_str = COLOR_DICT[action_result] + COLOR_DICT[action_chosen_by] + '{:<4}'.format(pa.get_mdp_state(next_z))
                mdp_traj_log += mdp_str

            z = next_z

        if pa.is_accepting_state(z):
            twtl_pass_count += 1

        # An episode has completed. The pa state must be reset while preserving the aug_mdp state.
        z, init_traj = pa.new_ep_state(z)

        if log:
            with open(tr_log_file, 'a') as log_file:
                log_file.write(str(trajectory_reward_log))
                log_file.write('\n')
            with open(mdp_log_file, 'a') as log_file:
                log_file.write(str(mdp_traj_log))
                log_file.write('\n')
            trajectory_reward_log = init_traj[:]
            init_mdp_traj = [pa.get_mdp_state(z) for z in init_traj]
            mdp_traj_log = ''
            for x in init_mdp_traj:
                mdp_traj_log += '{:<4}'.format(x)

            # write formatted q table to file
            # with open(q_table_file, 'w') as f:
            #     # time header
            #     f.write('{:<50}'.format(""))
            #     f.write(('{:<6}' * len(qtable)).format(*list(range(len(qtable)))))
            #     f.write('\n')

            #     for p,q_dict in qtable[0].items():
            #         for q,val in q_dict.items():
            #             f.write
            #     for i,_ in enumerate qtable:
            #         f.write('\n\nt = {}\n'.format(i))

    # print("TWTL success rate: {} / {} = {}".format(twtl_pass_count, episodes, twtl_pass_count/episodes))

    return pi

def test_policy(pi, pa, eps_unc, iters):

    print('Testing optimal policy with {} iterations'.format(iters))

    # z,t_init,init_traj = pa.initial_state_and_time(((None,None,'r7'), 0))
    # z,t_init,init_traj = pa.initial_state_and_time((('r7', (0,)), 0))
    z,t_init,init_traj = pa.initial_state_and_time()
    time_steps = pa.get_hrz()
    traj = []
    traj.extend(init_traj)

    # count TWTL satsifactions
    twtl_pass_count = 0
    stl_sat_count = 0
    total_count = 0

    for _ in range(iters):
        for t in range(t_init, time_steps):
            intended_z = pi[t][z]
            if intended_z == None:
                intended_z = pa.pi_eps_go[t][z]
            
            # take action
            next_z = pa.take_action(z, intended_z, eps_unc)

            total_count += 1
            if pa.sat(next_z):
                stl_sat_count += 1

            z = next_z

        if pa.is_accepting_state(z):
            twtl_pass_count += 1

        z, init_traj = pa.new_ep_state(z)


    print("TWTL success rate: {} / {} = {}".format(twtl_pass_count, iters, twtl_pass_count/iters))
    print("STL success rate: {}".format(stl_sat_count/total_count))




if __name__ == '__main__':
    start_time  = time.time()
    # custom_task = '[H^1 r46]^[0,10] * ([H^1 r57]^[0, 10] | [H^1 r24]^[0, 10])  * [H^1 Base1]^[0,10]' # '[H^1 r46]^[0,10] * ([H^1 r57]^[0, 10] | [H^1 r24]^[0, 10])  * [H^1 Base1]^[0,10]'
    custom_task = None
    ##### System Inputs for Data Prep. #####
    # ep_len = 21 # Episode length
    length = 4       # of rows
    width = 4       # of columns  8
    height = 1       # height set to 1 for 2D case
    ts_size = length * width

    # STL constraint
    stl_expr = 'G[0,20]F[0,2]((x>2)&(y>2))'
    # stl_expr = 'F[0,20]G[0,2]((x>2)&(x<3)&(y>2)&(y<3))'
    # stl = TmdpStl(stl_expr)
    # tau = stl.get_tau()
    # ep_len = int(stl.hrz())
    
    # Specify initial states and obstacles (row,column,altitude/height)
    init_state    = (0,0,0)
    
    pick_up_state = (3,3,0)
    delivery_state = (1,3,0)

    ##### System Inputs for Q-Learning #### 	#For debugging
    LEARN_FLAG = True  # False # If true learn a new Q table, if False load the previously found one
    sample_size = 10000 # Specify How Many samples to run

    num_episodes = 100000      # of episodes
    SHOW_EVERY = 5000       # Print out the info at every ... episode
    # LEARN_RATE = 0.1
    LEARN_RATE = 0.1
    DISCOUNT   = 0.999
    # EPS_DECAY  = 0.999995 #0.99989
    EPS_DECAY = 1
    epsilon    = 0.3# exploration trade-off
    eps_unc    = 0.03 # Uncertainity in actions, real uncertainnity in MDP
    eps_unc_learning = 0.05 # Overestimated uncertainity used in learning
    des_prob     = 0.86 # Minimum desired probability of satisfaction

    n_samples = 50 # Running the algorithm for different model based samples, 0 for model free learning

    prep_start_time = timeit.default_timer()

    pa = build_environment(length, width, height, init_state, pick_up_state, delivery_state, None, stl_expr)
    pa.prune_actions(eps_unc, des_prob)

    timer = timeit.default_timer()
    pi = Q_learning(pa, num_episodes, eps_unc, LEARN_RATE, DISCOUNT, EPS_DECAY, epsilon, n_samples)
    qlearning_time = timeit.default_timer() - timer
    print('learning time: {}'.format(qlearning_time))

    # test policy
    test_policy(pi, pa, eps_unc, 500)

    # generate trajectory
    # z,t_init,init_traj = pa.initial_state_and_time((('r7', (0,)), 0))
    # time_steps = pa.get_hrz()
    # traj = []
    # traj.extend(init_traj)
    # for t in range(t_init, time_steps):
    #     next_z = pi[t][z]
    #     traj.append(next_z)
    # print(traj)


    pi_file = '../output/optimal_policy.txt'
    with open(pi_file, 'w') as f:
        fmt = '{:<23} ||  ' + '{:>22}' * len(pi) + '\n'
        f.write(fmt.format('', *list(range(len(pi)))))
        for s in sorted(pi[0].keys()):
            s2 = [pi_t[s] for pi_t in pi]
            f.write(fmt.format(s, *s2))

    # for ind_p in range(len(pick_up_state)):
    #     for ind in range(len(delivery_state)):
            # [i_s_i, pa_i, pa_s_i, pa_t_i, pa2ts_i, energy_pa_i, pick_up_i, delivery_i,  pick_ups_i, deliveries_i, pa_g_nodes_i] = prep_for_learning(ep_len, m, n, h, init_states, pick_up_state[ind_p], delivery_state[ind], custom_flag, custom_task, tau)
            # i_s.append(i_s_i)
            # rewards_pa.append(rewards_pa_i)
            # pa.append(pa_i)
            # pa_s.append(pa_s_i)
            # pa_t.append(pa_t_i)
            # pa2ts.append(pa2ts_i)
            # energy_pa.append(energy_pa_i)
            # pick_up.append(pick_up_i)
            # delivery.append(delivery_i)
            # pick_ups.append(pick_ups_i)
            # deliveries.append(deliveries_i)
            # [possible_acts_time_included_not_pruned_i, possible_acts_time_included_pruned_i, possible_next_states_time_included_not_pruned_i, possible_next_states_time_included_pruned_i, act_num_i] = get_possible_actions(pa_g_nodes_i,energy_pa_i, pa2ts_i, pa_s_i, pa_t_i, ep_len, Pr_des, eps_unc, pick_up_i)
            # possible_acts_not_pruned.append(possible_acts_time_included_not_pruned_i)
            # possible_acts_pruned.append(possible_acts_time_included_pruned_i)
            # possible_next_states_not_pruned.append(possible_next_states_time_included_not_pruned_i)
            # possible_next_states_pruned.append(possible_next_states_time_included_pruned_i)
            # act_num.append(act_num_i)

    # prep_timecost =  timeit.default_timer() - prep_start_time
    # print('Total time for data prep. : ' + str(prep_timecost) + ' seconds \n')
