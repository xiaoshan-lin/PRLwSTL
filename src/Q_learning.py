import os
import numpy as np
import random

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

def Q_learning(pa, episodes, eps_unc, learn_rate, discount, eps_decay, epsilon, log=True):
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
    log : boolean
        Whether log files should be created
        default is True

    Returns
    -------
    dict
        The optimal policy pi as a dict of dicts. The outer is keyed by the time step
        and the inner is keyed by the Product MDP state. The value is the optimal next Product MDP state.
    """
    # Time complexity is O(tx+pt) = O(t(x+p))

    # Log state sequence and reward
    trajectory_reward_log = []
    mdp_traj_log = ''
    tr_log_file = os.path.join(this_file_path, '../output/trajectory_reward_log.txt')
    mdp_log_file = os.path.join(this_file_path, '../output/mdp_trajectory_log.txt')
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

    # Make an entry in q table for learning initial states and initialize pi
    if pa.is_STL_objective:
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
            # TODO: shouldn't this update based on probable_z as that was the "action"?
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

        z = pa.get_null_state(z)

        if pa.is_STL_objective:
            #FIXME: pi[0][z] could be None
            # Choose init state either randomly or by pi
            if np.random.uniform() < epsilon:   # Explore
                possible_init_zs = list(qtable[0][z].keys())
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

        else:
            # static rewards: Randomly choose adjacent state for beginning of next ep
            init_states = list(pa.g.neighbors(z))
            if init_states == []:
                raise RuntimeError('ERROR: No neighbors of final state? Actions not reversible?')
            init_z = random.choice(init_states)
            # Don't want any progress toward TWTL satisfaction on this transition
            init_z = pa.get_null_state(init_z)

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
    # Time complexity is O(t)

    print('Testing optimal policy with {} episodes'.format(iters))

    mdp_log_file = os.path.join(this_file_path, '../output/test_policy_trajectory_log.txt')
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
        rdeg = 0
        if pa.is_STL_objective:
            z_init = pi[0][z_null]
            init_traj = pa.get_new_ep_trajectory(z,z_init)
            mdp_sig = [pa.aug_mdp.sig_dict[x] for x in mdp_traj]
            rdeg = parser.rdegree(mdp_sig)
            if rdeg > 0:
                stl_sat_count += 1
            stl_rdeg_sum += rdeg
        else:
            # Choose random adjacent
            init_states = list(pa.g.neighbors(z))
            z_init = random.choice(init_states)
            z_init = pa.get_null_state(z_init)
            init_traj = [z_init]
        z = z_init


        mdp_traj = [pa.get_mdp_state(p) for p in init_traj]
        reward_sum += pa.reward(z_init)
        # for p in init_traj:
        #     reward_sum += pa.reward(p)


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
