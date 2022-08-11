

from pyTWTL import lomap
import pyTWTL.synthesis as synth
import networkx as nx
import math
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

class AugPa(lomap.Model):

    def __init__(self, aug_mdp, mdp_type, dfa, time_bound, width, height):
        # aug_mdp is an augmented mdp such as a tau-mdp or flag-mdp
        # dfa is generated from a twtl constraint
        # time_bound is both the time bound of the twtl task,W and the time horizon of the STL constraint

        lomap.Model.__init__(self, directed=True, multi=False)

        self.aug_mdp = aug_mdp
        self.mdp_type = mdp_type
        self.dfa = dfa
        self.time_bound = time_bound
        self.reward_cache = {}
        self.is_STL_objective = not (aug_mdp.name == 'Static Reward MDP')
        self.width = width
        self.height = height
        self.tau = self.aug_mdp.get_tau()
        #print(aug_mdp.g.nodes())
        #self.plot_graph(aug_mdp.g)

        #self.aug_---mdp.visualize()
        
        # generate
        # Following has O(xda*2^|AP|) worst case. O(xd) for looping through all PA states * O(a * 2^|AP|) for adjacent MDP and DFA states
        product_model = synth.ts_times_fsa(aug_mdp, dfa)
        self.init = product_model.init
        self.g = product_model.g
        self.final = product_model.final
        self.idx_to_action = {0:'stay',1:'E',-1:'W',-self.width:'N',self.width:'S',
                              -self.width-1:'NW',-self.width+1:'NE',self.width-1:'SW',self.width+1:'SE'}
        self.action_to_idx = {'stay':0,'E':1,'W':-1,'N':-self.width,'S':self.width,
                              'NW':-self.width-1,'NE':-self.width+1,'SW':self.width-1,'SE':self.width+1}
        if self.width < 3 and self.height < 3:
            self.correct_action_flag = True
            self.corrected_action_left = {'SW':'E','W':'NE'}
            self.corrected_action_right = {'E':'SW','NE':'W',}
        else:
            self.correct_action_flag = False

        # TODO: reset_init seems like a messy thing to do
        aug_mdp.reset_init()
        aug_mdp_init = list(aug_mdp.init.keys())[0]
        dfa_init = list(dfa.init.keys())[0]

        # May need to remove a certain aug mdp state
        to_remove = aug_mdp.get_state_to_remove()
        pa_to_remove = [p for p in self.get_states() if self.get_aug_mdp_state(p) == to_remove]
        self.g.remove_nodes_from(pa_to_remove)

        self.init = {(aug_mdp_init, dfa_init):1}

        # Generate set of null states
        self.null_states = self._gen_null_states()

        # allow caller to compute energy so it can be timed
        self.energy_dict = None

    def _gen_null_states(self):
        null_aug_mdp_states = set([self.aug_mdp.get_null_state(s) for s in self.aug_mdp.g.nodes()])
        null_pa_states = [(s, list(self.dfa.init.keys())[0]) for s in null_aug_mdp_states]
        for i,z in enumerate(null_pa_states):
            if z not in self.get_states():
                # Due to using label of s' in DFA update
                z = (z[0], list(self.dfa.init.keys())[0] + 1)
                if z not in self.get_states():
                    # if still an invalid state, something is wrong
                    raise Exception('Error: state not in Product MDP: {}'.format(z))
                null_pa_states[i] = z

        return null_pa_states

    def get_null_states(self):
        return self.null_states

    def get_aug_mdp_state(self, pa_state):
        return pa_state[0]

    def get_mdp_state(self, pa_state):
        aug_mdp_state = pa_state[0]
        return self.aug_mdp.get_mdp_state(aug_mdp_state)

    def get_dfa_state(self, pa_state):
        dfa_state = pa_state[1]
        return dfa_state

    def get_hrz(self):
        return self.aug_mdp.get_hrz()

    def get_states(self):
        return self.g.nodes()

    def get_tpa_state_size(self):
        return len(self.g.nodes()) * self.time_bound

    def compute_energy(self):

        # Decrease compute time significantly by computing energy over pa of simple mdp and dfa
        #   and then projecting to this pa

        mdp = self.aug_mdp.get_mdp()
        # Following is O(nda*2^|AP|)
        simple_pa = synth.ts_times_fsa(mdp, self.dfa)

        # make a virtual node as the end point using 0 for weight
        simple_pa.g.add_edges_from([(p, 'virtual', {'weight':0}) for p in simple_pa.final])

        # compute minimum path costs (energy)
        # NOTE: the simple_pa does not have weights. ts_times_fsa does not carry them over. ts_times_fsa must be reimplemented if weights on the PA are desired.
        #       I specify the 'weight' attribute anyways for future sake.
        # It seems that any node that cannot reach target is excluded from the returned dict. It does not raise an error as the docs state.
        # complexity of dijkstra using min-priority que is O((v+e)log(v)) with verticies v and edges e. This can be simplified to O(e*log(v)) if the graph is connected.
        # O(nad*log(nd)). With nd nodes, nad edges, because the product MDP uses the MDP transition function with edges n*a, applied d times.
        simple_energy_dict = nx.shortest_path_length(simple_pa.g, target='virtual', weight='weight')

        # project onto full PA
        energy_dict = {}
        for p in self.get_states():
            mdp_s = self.get_mdp_state(p)
            dfa_s = self.get_dfa_state(p)
            try:
                energy_dict[p] = simple_energy_dict[(mdp_s, dfa_s)]
            except KeyError:
                energy_dict[p] = float('inf')
        self.energy_dict = energy_dict
        nx.set_node_attributes(self.g, energy_dict, name='energy')

    def get_energy(self, pa_state):
        return self.energy_dict[pa_state]

    def plot_graph(self, graph):
        pygraphviz_g = nx.nx_agraph.to_agraph(graph)
        pygraphviz_g.layout('dot', args='-Nfontsize=30 -Nwidth="3" -Nheight="1" -Nmargin=0 -Gfontsize=15')
        pygraphviz_g.draw('example.png', format='png')
        img = plt.imread('example.png')
        plt.imshow(img)
        plt.show()

    def prune_actions(self, eps_uncertainty, des_prob):
        # Time complexity is O(txa^2) 
        # Loop through all time product states, all neighbors, all low probability transitions.
        # In the current implementation, there cannot be more low prob  transitions than actions

        # TODO actually use actions and not next state

        if self.energy_dict == None:
            raise Exception("Please call compute_energy before prune_actions")

        ep_len = self.time_bound
        # initialize actions of time product MDP
        pruned_states = [nx.convert.to_dict_of_lists(self.g) for _ in range(ep_len)]
        pruned_actions = {t:{p:[] for p in self.g.nodes()} for t in range(ep_len)}

        # create set of non-accepting states
        accepting_states = self.final
        non_accepting_states = list(pruned_states[0].keys())
        for s in accepting_states:
            non_accepting_states.remove(s)
            for t in range(ep_len):
                pruned_actions[t][s] = [self.states_to_action(s,q) for q in pruned_states[t][s]]           
        
        for p in tqdm(non_accepting_states):
            for t in range(ep_len):
                pi_eps_go_states = []
                d_eps_min = float('inf')
                # Create copy to avoid removing from the same list being iterated
                next_ps = pruned_states[t][p][:]
                for next_p in next_ps:
                    # Find all possible epsilon stochastic transitions resulting from this edge
                    low_prob_neighbors = self.get_low_prob_neighbors(p,next_p)
                    neighbors = low_prob_neighbors + [next_p]
                    # dmax = max([self.energy_dict[n] for n in neighbors]) + 1 # TODO: FIXME
                    dmax = max([self.energy_dict[n] for n in neighbors]) # THIS IS CORRECT
                    k = ep_len - t
                    imax = int(math.floor((k - 1 - dmax) / 2))

                    sat_prob = 0
                    for i in range(imax + 1):
                        part_1 = math.factorial(k - 1) / (math.factorial(k - 1 - i) * math.factorial(i))
                        part_2 = eps_uncertainty ** i
                        # part_3 = (1 - eps_uncertainty) ** (k - i) # TODO: FIXME
                        part_3 = (1 - eps_uncertainty) ** (k - 1 - i) # THIS IS CORRECT
                        sat_prob += part_1 * part_2 * part_3

                    if imax < 0 or sat_prob < des_prob:
                        # prune this action
                        pruned_states[t][p].remove(next_p)

                    # track minimum epsilon stocastic transition distance for the case that the action set ends empty
                    d_eps = self.energy_dict[next_p]
                    if d_eps < d_eps_min:
                        d_eps_min = d_eps
                        pi_eps_go_states = [next_p]
                    elif d_eps == d_eps_min:
                        pi_eps_go_states.append(next_p)
                        
                #print([t,p])
                #print(pruned_states[t][p])
                if pruned_states[t][p] == []:
                    # record state signifying action minimizing d-epsilon-min
                    # In this scenario with one state for each action with p > 1 - eps, 
                    #   a state-state edge can represent an action
                    pruned_states[t][p] = pi_eps_go_states
                    #print(pi_eps_go_states)
                
                pruned_actions[t][p] = [self.states_to_action(p,q) for q in pruned_states[t][p]]
                #print(pruned_actions[t][p]) 
                #print('---')
        #print(pruned_states)
        self.pruned_states = pruned_states
        self.pruned_actions = pruned_actions
        #print( self.g.nodes())
        #print(self.pruned_actions)

    def states_to_action(self, s1, s2): 
        match self.mdp_type:
            case 'static rewards':
                idx1 = int(s1[0][1:])
                idx2 = int(s2[0][1:])
            case 'tau-MDP':
                idx1 = int(s2[0][self.tau-2][1:])
                idx2 = int(s2[0][self.tau-1][1:])
            case 'flag-MDP':
                idx1 = int(s1[0][0][1:])
                idx2 = int(s2[0][0][1:])
        action = self.idx_to_action[idx2-idx1]
        if self.correct_action_flag:
            if action in self.corrected_action_left and (idx1==0 or idx1==2):
                action = self.corrected_action_left[action]
                print([s1,s2,action])
            elif action in self.corrected_action_right and (idx1==1 or idx1==3):
                action = self.corrected_action_right[action]
                print([s1,s2,action])           
        return action       

    def take_action(self, s, a, uncertainty):
        # Action is being defined as a state-state transition. 
        #   Possibly use discrete actions in the future in the case that multiple states can
        #   result from a single action with significant probability.
        
        # for verifying next_state
        # next_state = [i for i in self.pruned_states[t][s] if int(i[0][1:])==next_idx][0]
        match self.mdp_type:
            case 'static rewards':
                cur_idx = int(s[0][1:])
                next_idx = cur_idx + self.action_to_idx[a]
                next_aug_mdp_s = 'r{}'.format(next_idx)
            case 'tau-MDP':
                cur_idx = int(s[0][self.tau-1][1:])
                next_idx = cur_idx + self.action_to_idx[a]
                next_mdp_s = 'r{}'.format(next_idx)
                next_aug_mdp_s = s[0][1:]+(next_mdp_s,)
            case 'flag-MDP':
                cur_idx = int(s[0][0][1:])
                next_idx = cur_idx + self.action_to_idx[a]
                next_mdp_s = 'r{}'.format(next_idx)
                flags = s[0][1]
                next_flags = self.aug_mdp.fmdp_stl.flag_update(flags, next_mdp_s)
                next_aug_mdp_s = (next_mdp_s,next_flags)

        ts_prop = self.aug_mdp.g.nodes[next_aug_mdp_s].get('prop',set())
        fsa_next_state = self.dfa.next_states_of_fsa(s[1], ts_prop)[0]
        next_s = (next_aug_mdp_s, fsa_next_state)

        if np.random.uniform() > uncertainty:
            return next_s
        else:
            # Choose next state from possible low probability states
            low_prob_states = self.get_low_prob_neighbors(s,next_s)
            if low_prob_states == []:
                # if no low prob states exist, action must be stay and s2 is the only option
                return next_s

            # Choose from low probability states
            next_s = random.choice(low_prob_states)
            return next_s


    def get_low_prob_neighbors(self, s1, s2):
        # TODO: put some of this in an mdp class
        # region_to_xy = self.aug_mdp.sig_dict
        # TODO should be more generalized than using sig_dict
        region_to_xy = {r:(d['x'],d['y']) for r,d in self.aug_mdp.sig_dict.items()}
        neighbors = self.g.neighbors(s1)
        xy_to_pa = {region_to_xy[self.get_mdp_state(pa_s)]:pa_s for pa_s in neighbors}

        mdp1 = self.get_mdp_state(s1)
        # mdp2 = self.get_mdp_state(s2)

        x,y = region_to_xy[mdp1]

        # Map directions to next states
        # This may include "out of bounds" states
        adj_xy_dict = {
            'N': (x-1,y),
            'NE':(x-1,y+1),
            'E': (x,y+1),
            'SE':(x+1,y+1),
            'S': (x+1,y),
            'SW':(x+1,y-1),
            'W': (x,y-1),
            'NW':(x-1,y-1),
            'stay':(x,y)
        }

        # convert xy to pa state and drop invalid
        adj_dict = {}
        for k in adj_xy_dict:
            try:
                adj_dict[k] = xy_to_pa[adj_xy_dict[k]]
            except KeyError:
                pass

        adj_dict_inv = {adj_dict[k]:k for k in adj_dict}
        dir_list = ['N','NE','E','SE','S','SW','W','NW']

        chosen_dir = adj_dict_inv[s2]
        if chosen_dir == 'stay':
            # No low probability states for this action
            return []

        low_prob_states = []
        chosen_idx = dir_list.index(chosen_dir)
        for i in [-1, 1]:
            alt_dir = dir_list[(chosen_idx + i) % len(dir_list)]
            try:
                low_prob_states.append(adj_dict[alt_dir])
            except:
                # alt_dir is out of the environment
                pass
        low_prob_states.append(adj_dict['stay'])  # stay

        return low_prob_states

    def initial_state_and_time(self, init_pa_state = None):
        # assume agent has remained in initial mdp state for timesteps 0 thru tau - 2
        # also give initial time that rewards are summed over in the q-learning problem formulation
        if init_pa_state == None:
            z = list(self.init.keys())[0]
        else:
            z = init_pa_state
            if z not in self.get_states():
                raise Exception("invalid pa state: {}".format(z))
        tau = self.aug_mdp.get_init_tau()
        init_traj = [z]

        '''for _ in range(1,tau):
            mdp_s = self.get_mdp_state(z)
            neighbors = self.g.neighbors(z)
            # choose next z with same mdp state
            z = next(iter([next_z for next_z in neighbors if self.get_mdp_state(next_z) == mdp_s]))
            init_traj.append(z)'''
        
        t_init = 0
        return z, t_init, init_traj

    def reward(self, pa_s, beta = 2):
        aug_mdp_s = pa_s[0]
        try:
            rew = self.reward_cache[(aug_mdp_s,beta)]
        except KeyError:
            if not self.aug_mdp.is_state(aug_mdp_s):
                raise Exception("Invalid augmented mdp state!")
            rew = self.aug_mdp.reward(aug_mdp_s, beta)
            self.reward_cache[(aug_mdp_s,beta)] = rew
        return rew

    # def new_ep_state(self, last_pa_state):
    #     # reset DFA state
    #     aug_mdp_s = last_pa_state[0]
    #     aug_mdp_init_s = self.aug_mdp.new_ep_state(aug_mdp_s)
    #     dfa_init_s = self.dfa.init.keys()[0]
    #     new_pa_s = (aug_mdp_init_s, dfa_init_s)
    #     if new_pa_s not in self.get_states():
    #         # This is the case if using the label of s' in the DFA update rather than the label of s.
    #         # MDP states with a label corresponding to the first hold will not have a PA state with the inital DFA state.
    #         new_pa_s = (aug_mdp_init_s, dfa_init_s + 1)
    #     z, _, init_traj = self.initial_state_and_time(new_pa_s)

    #     return z, init_traj

    def get_null_state(self, pa_s):
        aug_mdp_s = self.get_aug_mdp_state(pa_s)
        null_aug_mdp_s = self.aug_mdp.get_null_state(aug_mdp_s)
        null_pa_s = (null_aug_mdp_s, list(self.dfa.init.keys())[0])
        if null_pa_s not in self.get_states():
            # Due to using label of s' in DFA update
            null_pa_s = (null_aug_mdp_s, list(self.dfa.init.keys())[0] + 1)
            if null_pa_s not in self.get_states():
                raise Exception('Error: invalid null state: {}'.format(null_pa_s))
        return null_pa_s

    def is_accepting_state(self, pa_s):
        dfa_state = self.get_dfa_state(pa_s)
        return dfa_state in self.dfa.final

    def sat(self, pa_s):
        aug_mdp_s = self.get_aug_mdp_state(pa_s)
        return self.aug_mdp.sat(aug_mdp_s)

    def gen_new_ep_states(self):
        # Time complexity is O(x*n^r)
        # Generates a dictionary that maps each pa state (at end of ep) to possible pa states at t = tau-1, and each of those 
        #   PA states to possible initial trajectories that could lead to that
        # Each new episode starts with a state adjacent to the last state of the previous episode

        # Nothing to do for non STL objective
        if not self.is_STL_objective:
            return

        if self.mdp_type == 'flag-MDP':
            new_ep_dict = {p:{} for p in self.get_null_states()}
            for p in new_ep_dict:
                # choose init state of new episode from the neighbor of the last state
                new_ep_dict[p] = {q:[] for q in self.get_null_states() if q[0][0] in self.aug_mdp.mdp.g.neighbors(p[0][0])}
            return new_ep_dict
            
        new_ep_dict = {}
        def new_ep_states_recurse(pa_s, tau, t = 0, temp_dict = None, hist = None):

            # Assumes that pa_s is the state chosen at t = 0 TODO: is this still correct?
            if temp_dict == None:
                temp_dict = {}
            if hist == None:
                hist = []
            else:
                hist.append(pa_s)

            try:
                neighbors = self.pruned_states[t][pa_s]
            except KeyError:
                pa_s = (pa_s[0], list(self.dfa.init.keys())[0] + 1)
                neighbors = self.pruned_states[t][pa_s]

            # for eg static rewards
            if tau == 1:
                return {n: [[n]] for n in neighbors}

            if t == tau-1:
                for n in neighbors:
                    if n in temp_dict:
                        # temp_dict[n].append(hist + [n])
                        pass # TODO: Find a less memory intensive method. Take first history until then.
                    else:
                        temp_dict[n] = [hist + [n]]
                hist.pop()
                return temp_dict
            
            for n in neighbors:
                temp_dict = new_ep_states_recurse(n, tau, t + 1, temp_dict, hist)
            if t != 0:
                hist.pop()
            return temp_dict

        tau = self.aug_mdp.get_tau()
        # if tau < 2:
        #     raise Exception("Tau < 2 not supported in initial state selection")                    

        for pa_s in tqdm(self.get_states()):
            if pa_s not in new_ep_dict:
                aug_mdp_s = self.get_aug_mdp_state(pa_s)
                null_aug_mdp_s = self.aug_mdp.get_null_state(aug_mdp_s)
                null_pa_s = (null_aug_mdp_s, list(self.dfa.init.keys())[0])
                if null_pa_s not in new_ep_dict:
                    new_ep_dict[null_pa_s] = new_ep_states_recurse(null_pa_s, tau)
                    if new_ep_dict[null_pa_s] == {}:
                        # TODO: 
                        raise Exception('\nError: Augmented Product MDP state {} at t=0 can not reach a state at t={} '.format(null_pa_s, tau) 
                                + 'on the Pruned Augmented Time-Product MDP. This is most likely due to all potential actions being pruned. '
                                + 'It is assumed that all states at t=0 can reach a state at t=tau so that, in the case of an STL objective, '
                                + 'potential initial trajectories can be computed.\n'
                                + 'This can be resolved by either (1) reducing the over estimated action uncertainty, (2) reducing the desired '
                                + 'satisfaction probability (3) increasing the time horizon, or (4) modifying the constraint mission such that '
                                + 'it can be completed in fewer time steps.')
                new_ep_dict[pa_s] = new_ep_dict[null_pa_s]
                # TODO maybe only key on null_pa_s to save ram for large state spaces, if it is indeed copying here

        self.new_ep_dict = new_ep_dict

    def get_new_ep_states(self, pa_s):
        if not self.is_STL_objective:
            raise RuntimeError('This function should not be called for non STL objective.')
        new_ep_states = list(self.new_ep_dict[pa_s].keys())
        return new_ep_states

    def get_new_ep_trajectory(self, last_pa_s, init_pa_s):
        if not self.is_STL_objective:
            raise RuntimeError('This function should not be called for non STL objective.')
        new_ep_trajs = self.new_ep_dict[last_pa_s][init_pa_s]
        selection_idx = np.random.choice(len(new_ep_trajs))
        selection = new_ep_trajs[selection_idx]
        return selection

                
