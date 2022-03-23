
from __future__ import division # dividing two integers produces a float

import create_environment as ce
import timeit, time
import networkx as nx
import matplotlib.pyplot as plt
import yaml
import os

from pyTWTL.twtl_to_dfa import twtl_to_dfa
from pyTWTL import lomap
from pyTWTL import synthesis as synth

import copy
import numpy as np
import random
from tmdp_stl import Tmdp
from product_automaton import AugPa
from fmdp_stl import Fmdp
from static_reward_mdp import StaticRewardMdp
from STL import STL


NORMAL_COLOR = '\033[0m'

COLOR_DICT = {
    'pi epsilon go' : '\033[38;5;3m',   # yellow
    'explore'       : '\033[38;5;4m',   # blue
    'exploit'       : '\033[38;5;2m',   # green
    'intended'      : '\033[49m',       # no highlight
    'unintended'    : '\033[48;5;1m'    # red highlight
}

this_file_path = os.path.dirname(os.path.abspath(__file__))


def build_environment(env_cfg, twtl_cfg, mdp_type, reward_cfg):
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

    pickup = twtl_cfg['pickup']
    delivery = twtl_cfg['delivery']
    custom_task = twtl_cfg['custom task']

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
        hrz = reward_cfg['time horizon']
        reward_dict = reward_cfg['reward dict']
        aug_mdp = StaticRewardMdp(ts, hrz, state_to_pos, reward_dict)
    else:
        raise Exception("invalid AUG_MDP_TYPE")
    aug_mdp_timecost = timeit.default_timer() - aug_mdp_timer

    # TODO: check that this is the correct episode length
    ep_len = aug_mdp.get_hrz()


    # =================================
    # DFA Creation
    # =================================
    dfa_start_time = timeit.default_timer()
    pick_up_reg = xy_to_region(*pickup)
    delivery_reg = xy_to_region(*delivery)
    pick_up_str  = str(pick_up_reg)
    delivery_str = str(delivery_reg)
    if custom_task != 'None':
        phi = custom_task
        raise Exception("Error: Custom task time bound compatability checking is not yet implemented!")
    else:
        tf1 = int((ep_len-1)/2) # time bound
        tf2 = int(ep_len) - tf1 - 1
        phi = '[H^1 r' + pick_up_str + ']^[0, ' +  str(tf1) + '] * [H^1 r' + delivery_str + ']^[0,' + str(tf2) + ']'  # Construc the task according to pickup/delivery )^[0, ' + tf + ']'
    # _, dfa_inf, _ = twtl.translate(phi, kind=DFAType.Infinity, norm=True) # states and sim. time ex. phi = '([H^1 r47]^[0, 30] * [H^1 r31]^[0, 30])^[0, 30]' 
    out = twtl_to_dfa(phi, kind='infinity', norm=True)
    dfa_inf = out['infinity']
    dfa_timecost =  timeit.default_timer() - dfa_start_time # DFAType.Normal for normal, DFAType.Infinity for relaxed

    # add self edge to accepting state
    # TODO probably don't hardcode the input set. see dfa_inf.alphabet
    dfa_inf.g.add_edge(4,4, **{'guard': '(else)', 'input':set([0,1,2,3]), 'label':'(else)', 'weight':0})
    
    # plt.subplot()
    # nx.draw(dfa_inf.g, with_labels=True)
    # plt.show()

    # =================================
    # Augmented Product MDP Creation
    # =================================
    pa_start_time = timeit.default_timer()
    pa_or = AugPa(aug_mdp, dfa_inf, ep_len)
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
    print('##### PICK-UP and DELIVERY MISSION #####' + "\n")
    print('Initial Location  : ' + str(init_state) + ' <---> Region ' + str(init_state_num))
    print('Pick-up Location  : ' + str(pickup) + ' <---> Region ' + pick_up_str)
    print('Delivery Location : ' + str(delivery) + ' <---> Region ' + delivery_str)
    # print('Reward Locations  : ' + str(rewards) + ' <---> Regions ' + str(rewards_ts_indexes) + "\n")
    print('State Matrix : ')
    print(state_mat)
    print("\n")
    print('Mission Duration  : ' + str(ep_len) + ' time steps')
    print('TWTL Task : ' + phi + "\n")
    print('Time PA state size: {}\n'.format(pa.get_tpa_state_size()))
    print('Time Cost:')
    print('TS creation time (s):            {:<7}'.format(ts_timecost))
    print('Augmented MDP creation time (s): {:<7}'.format(aug_mdp_timecost))
    # print('			TS created in ' + str(ts_timecost) + ' seconds')
    print('DFA creation time (s):           {:<7}'.format(dfa_timecost))
    print('PA creation time (s):            {:<7}'.format(pa_timecost))
    print('PA energy calculation time (s):  {:<7}'.format(energy_timecost))

    return pa

def Q_learning(pa, episodes, eps_unc, learn_rate, discount, eps_decay, epsilon):
    """
    Find the optimal policy using Q-learning

    Parameters
    ----------
    pa : AugPa
        The Augmented Product MDP
    episodes : int
        The number of episodes of learning
    eps_unc : float
        The real action uncertainty. This is the probability of an unintended transition.
    learn_rate : float
        The learning rate used in the Q update function
    discount : float
        The future value discount used in the Q update function
    eps_decay : float
        The decay rate of the exploration probability. 
        (inital prob) * decay^(episodes - 1) = (final prob)
    epsilon : float
        The initial exploration probability

    Returns
    -------
    dict
        The optimal policy pi as a dict of dicts. The outer is keyed by the time step
        and the inner is keyed by the Product MDP state. The value is the optimal next Product MDP state.
    """

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
    ep_rew_sum = 0
    ep_rewards = np.zeros(episodes)

    # initial state,time
    z,t_init,init_traj = pa.initial_state_and_time()

    # initialize Q table
    init_val = 0
    time_steps = pa.get_hrz()
    qtable = {i:{} for i in range(t_init, time_steps)}
    for i in qtable:
        qtable[i] = {p:{} for p in pa.get_states()}
        for p in qtable[i]:
            qtable[i][p] = {q:init_val + np.random.normal(0,0.0001) for q in pa.g.neighbors(p)}

    # initialize optimal policy pi on pruned time product automaton
    pi = {t:{} for t in qtable}
    for t in pi:
        for p in pa.pruned_time_actions[t]:
            if pa.pruned_time_actions[t][p] != []:
                # initialize with a neighbor in the pruned space
                pi[t][p] = max(pa.pruned_time_actions[t][p], key=qtable[t][p].get)
            else:
                # Empty action set. No actions available in the pruned time pa.
                pi[t][p] = None
        # pi[t] = {p:max(qtable[t][p], key=qtable[t][p].get) for p in pa.pruned_time_actions[t]}

    # Make an entry in q table for learning initial states and initialize pi
    qtable[0] = {p:{} for p in pa.get_null_states()}
    pi[0] = {}
    for p in qtable[0]:
        qtable[0][p] = {q:init_val + np.random.normal(0,0.0001) for q in pa.get_new_ep_states(p)}
        pi[0][p] = max(qtable[0][p], key=qtable[0][p].get)

    if log:
        trajectory_reward_log.extend(init_traj)
        init_mdp_traj = [pa.get_mdp_state(z) for z in init_traj]
        for x in init_mdp_traj:
            mdp_traj_log += '{:<4}'.format(x)
    # z = pa.init.keys()[0]

    # Loop for number of training episodes
    for ep in range(episodes):
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
            if t+1 == time_steps:
                max_future_q = 0
            else:
                future_qs = qtable[t+1][next_z]
                max_future_q = max(future_qs.values())

            # Update q value
            new_q = (1 - learn_rate) * cur_q + learn_rate * (reward + discount * max_future_q)
            qtable[t][z][next_z] = new_q

            # Update optimal policy
            if next_states != []:
                pi[t][z] = max(next_states, key=qtable[t][z].get)

            # track sum of rewards
            ep_rew_sum += reward

            if log:
                trajectory_reward_log.append(next_z)
                mdp_str = COLOR_DICT[action_result] + COLOR_DICT[action_chosen_by] + '{:<4}'.format(pa.get_mdp_state(next_z))
                mdp_traj_log += mdp_str

            z = next_z

        epsilon = epsilon * eps_decay

        if pa.is_accepting_state(z):
            twtl_pass_count += 1

        ep_rewards[ep] = ep_rew_sum
        ep_rew_sum = 0

        # choose initial trajectory for next episode
        # Choose init state either randomly or by pi
        z = pa.get_null_state(z)
        if np.random.uniform() < epsilon:   # Explore
            possible_init_zs = qtable[0][z].keys()
            init_z = random.choice(possible_init_zs)
            action_chosen_by = "explore"
        else:                               # Exploit
            init_z = pi[0][z]
            action_chosen_by = "exploit"

        init_traj = pa.get_new_ep_trajectory(z, init_z)

        # Update qtable and optimal policy
        reward = pa.reward(init_z)
        cur_q = qtable[0][z][init_z]
        future_qs = qtable[t_init][init_z]
        max_future_q = max(future_qs.values())
        new_q = (1 - learn_rate) * cur_q + learn_rate * (reward + discount * max_future_q)
        qtable[0][z][init_z] = new_q
        pi[0][z] = max(qtable[0][z], key=qtable[0][z].get)

        z = init_z

        if log:
            with open(tr_log_file, 'a') as log_file:
                log_file.write(str(trajectory_reward_log))
                log_file.write('\n')
            with open(mdp_log_file, 'a') as log_file:
                log_file.write(str(mdp_traj_log))
                log_file.write('\n')

            trajectory_reward_log = init_traj[:]
            init_mdp_traj = [pa.get_mdp_state(p) for p in init_traj]
            mdp_traj_log = ''
            for x in init_mdp_traj:
                mdp_traj_log += '{:<4}'.format(x)


    # print("TWTL success rate: {} / {} = {}".format(twtl_pass_count, episodes, twtl_pass_count/episodes))

    # plt.scatter(range(len(ep_rewards)), ep_rewards, alpha=0.3)
    # plt.xlabel('Episode')
    # plt.ylabel('Sum of rewards')
    # plt.show()


    return pi

def test_policy(pi, pa, stl_expr, eps_unc, iters, mdp_type):
    """
    Test a policy for a certian number of episodes and print 
        * The constraint mission success rate, 
        * The average sum of rewards for each episode
        * The objective (STL) mission success rate (if applicable)
        * The average robustness degree of each episode (if applicable)
    
    Parameters
    ----------
    pi : dict
        A policy as a dict of dicts. The outer is keyed by the time step
        and the inner is keyed by the Product MDP state. The value is an adjacent Product MDP state.
    pa : AugPA
        The Augmented Product MDP
    stl_expr : string
        The STL expression that represents the objective TODO: make this optional for the case of static rewards
    eps_unc : float
        The real action uncertainty. This is the probability of an unintended transition.
    iters : int
        The number of episodes to test over
    mdp_type : string
        The MDP augmentation type. Either 'static rewards', 'flag-MDP', or 'tau-MDP'

    Returns
    -------
    float
        The STL satisfaction rate TODO: return this only when applicable
    float
        The TWTL satisfaction rate
    
    """

    print('Testing optimal policy with {} episodes'.format(iters))

    mdp_log_file = '../output/test_policy_trajectory_log.txt'
    open(mdp_log_file, 'w').close() # clear file
    log = True

    # z,t_init,init_traj = pa.initial_state_and_time(((None,None,'r7'), 0))
    # z,t_init,init_traj = pa.initial_state_and_time((('r7', (0,)), 0))
    z,t_init,init_traj = pa.initial_state_and_time()
    time_steps = pa.get_hrz()
    # traj = []
    # traj.extend(init_traj)

    if log:
        mdp_traj_str = ''
        mdp_traj_log = []
        init_mdp_traj = [pa.get_mdp_state(z) for z in init_traj]
        for x in init_mdp_traj:
            mdp_traj_str += '{:<4}'.format(x)


    # count TWTL satsifactions
    twtl_pass_count = 0

    # Count STL satisfactions and avg robustness
    parser = STL(stl_expr)
    mdp_traj = [pa.get_mdp_state(z) for z in init_traj]
    stl_sat_count = 0
    stl_rdeg_sum = 0

    # count sum of rewards
    reward_sum = 0

    for _ in range(iters):
        for t in range(t_init, time_steps):
            intended_z = pi[t][z]
            if intended_z == None:
                intended_z = pa.pi_eps_go[t][z]
                action_chosen_by = "pi epsilon go"
            else:
                action_chosen_by = 'exploit'

            # take action
            next_z = pa.take_action(z, intended_z, eps_unc)
            action_result = 'intended' if next_z == intended_z else 'unintended'

            if log:
                mdp_str = COLOR_DICT[action_result] + COLOR_DICT[action_chosen_by] + '{:<4}'.format(pa.get_mdp_state(next_z))
                mdp_traj_str += mdp_str

            z = next_z

            mdp_traj.append(pa.get_mdp_state(next_z))
            reward_sum += pa.reward(next_z)

        if pa.is_accepting_state(z):
            twtl_pass_count += 1

        z_null = pa.get_null_state(z)
        z_init = pi[0][z_null]
        init_traj = pa.get_new_ep_trajectory(z,z_init)
        z = z_init

        mdp_sig = [pa.aug_mdp.sig_dict[x] for x in mdp_traj]
        rdeg = parser.rdegree(mdp_sig)
        if rdeg > 0:
            stl_sat_count += 1
        stl_rdeg_sum += rdeg

        mdp_traj = [pa.get_mdp_state(p) for p in init_traj]
        for p in init_traj:
            reward_sum += pa.reward(p)


        if log:
            mdp_traj_str += NORMAL_COLOR + '| {:>6}'.format(rdeg)
            mdp_traj_log.append(mdp_traj_str)
            mdp_traj_str = ''
            for pa_s in init_traj:
                mdp_s = pa.get_mdp_state(pa_s)
                mdp_traj_str += '{:<4}'.format(mdp_s)

    if log:
        with open(mdp_log_file, 'a') as log_file:
            for line in mdp_traj_log:
                log_file.write(line)
                log_file.write('\n')

    twtl_sat_rate = twtl_pass_count/iters
    stl_sat_rate = stl_sat_count/iters
    print("TWTL mission success: {} / {} = {}".format(twtl_pass_count, iters, twtl_pass_count/iters))
    print("Avg episode sum of rewards: {}".format(reward_sum/iters))
    if mdp_type != 'static rewards':
        print("STL mission success: {} / {} = {}".format(stl_sat_count, iters, stl_sat_count/iters))
        print("Avg robustness degree: {}".format(stl_rdeg_sum/iters))

    return stl_sat_rate, twtl_sat_rate

# def test_gen_time(iters = 5):
#     times = []
#     for i in range(iters):
#         start = timeit.default_timer()

#         pa = build_environment(length, width, height, INIT_STATE, PICKUP_STATE, DELIVERY_STATE, None, stl_expr)

#         # prune_start = timeit.default_timer()
#         pa.prune_actions(eps_unc, des_prob)
#         # prune_end = gen_start = timeit.default_timer()
#         # print('Time PA action pruning time (s): {}'.format(prune_end - prune_start))
#         pa.gen_new_ep_states()
#         # gen_end = timeit.default_timer()

#         end = timeit.default_timer()
#         times.append(end - start)
#     print('Average environment creation time: {}'.format(np.mean(times)))


def main():
    """
    Main function. Read in configuration values, construct the Pruned Time-Product MDP, 
    find the optimal policy, and test the optimal policy.
    """

    # Load default config
    my_path = os.path.dirname(os.path.abspath(__file__))
    def_cfg_rel_path = '../configs/default.yaml'
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

    # test_gen_time()

    # ==== Construct the Pruned Time-Product MDP ====
    prep_start_time = timeit.default_timer()

    # Construct the Product MDP
    pa = build_environment(env_cfg, twtl_cfg, mdp_type, reward_cfg)
    # Prune it at each time step
    prune_start = timeit.default_timer()
    pa.prune_actions(eps_unc_learning, des_prob)
    prune_end = gen_start = timeit.default_timer()
    print('Time PA action pruning time (s): {}'.format(prune_end - prune_start))
    pa.gen_new_ep_states()
    gen_end = timeit.default_timer()

    prep_end_time = timeit.default_timer()

    print('New ep/traj generation time (s): {}'.format(gen_end - gen_start))
    print('')
    print('Total environment creation time: {}'.format(prep_end_time - prep_start_time))
    print('')

    # ==== Find the optimal policy ====
    print('learning with {} episodes'.format(num_episodes))
    timer = timeit.default_timer()
    pi = Q_learning(pa, num_episodes, eps_unc, learn_rate, discount, explore_prob_decay, explore_prob_start)
    qlearning_time = timeit.default_timer() - timer
    print('learning time: {} seconds'.format(qlearning_time))

    # ==== test policy ====
    stl_expr = config['aug-MDP rewards']['STL expression']
    test_policy(pi, pa, stl_expr, eps_unc, 500, mdp_type)




if __name__ == '__main__':
    main()

