from __future__ import division

import lomap
import synthesis as synth
import networkx as nx
import math
import numpy as np
import random

class AugPa(lomap.Model):

    def __init__(self, aug_mdp, dfa, time_bound):
        # aug_mdp is an augmented mdp such as a tau-mdp or flag-mdp
        # dfa is generated from a twtl constraint
        # time_bound is both the time bound of the twtl task, and the time horizon of the STL constraint

        lomap.Model.__init__(self, directed=True, multi=False)

        self.aug_mdp = aug_mdp
        self.dfa = dfa
        self.time_bound = time_bound

        # generate
        synth.ts_times_fsa(aug_mdp, dfa, self)
        aug_mdp.reset_init()
        aug_mdp_init = aug_mdp.init.keys()[0]
        dfa_init = dfa.init.keys()[0]

        # May need to remove a certain aug mdp state
        to_remove = aug_mdp.get_state_to_remove()
        pa_to_remove = [p for p in self.get_states() if self.get_aug_mdp_state(p) == to_remove]
        self.g.remove_nodes_from(pa_to_remove)

        self.init = {(aug_mdp_init, dfa_init):1}

        # allow caller to compute energy so it can be timed
        self.energy_dict = None

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

    def compute_energy(self):

        #increase speed significantly by computing energy over pa of simple mdp and dfa
        #   and then projecting to this pa

        mdp = self.aug_mdp.get_mdp()
        simple_pa = synth.ts_times_fsa(mdp, self.dfa)

        # compute_energy wants a "new_weight" attribute. Just weight all transitions the same.
        new_weight_dict = {edge:1 for edge in simple_pa.g.edges()}
        nx.set_edge_attributes(simple_pa.g, 'new_weight', new_weight_dict)

        synth.compute_energy(simple_pa)
        mdp_energy_dict = nx.get_node_attributes(simple_pa.g, 'energy')
        energy_dict = {}
        for p in self.get_states():
            mdp_s = self.get_mdp_state(p)
            dfa_s = self.get_dfa_state(p)
            energy_dict[p] = mdp_energy_dict[(mdp_s, dfa_s)]
        self.energy_dict = energy_dict
        nx.set_node_attributes(self.g, 'energy', energy_dict)

    def get_energy(self, pa_state):
        return self.energy_dict[pa_state]

    def prune_actions(self, eps_uncertainty, des_prob):

        # TODO actually use actions and not next state

        if self.energy_dict == None:
            raise Exception("Please call compute_energy before prune_actions")


        ep_len = self.time_bound
        # initialize actions of time product MDP
        actions = [nx.convert.to_dict_of_lists(self.g) for _ in range(ep_len)]

        pi_eps_go = {t:{} for t in range(ep_len)}

        # create set of non-accepting states
        accepting_states = self.final
        non_accepting_states = actions[0].keys()
        for s in accepting_states:
            non_accepting_states.remove(s)
        
        for p in non_accepting_states:
            for t in range(ep_len):
                pi_eps_go_state = None
                d_eps_min = float('inf')
                # Create copy to avoid removing from the same list being iterated
                next_ps = actions[t][p][:]
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
                        actions[t][p].remove(next_p)

                    # track minimum epsilon stocastic transition distance for the case that the action set ends empty
                    d_eps = self.energy_dict[next_p]
                    if d_eps < d_eps_min:
                        d_eps_min = d_eps
                        pi_eps_go_state = next_p

                if actions[t][p] == []:
                    # record state signifying action minimizing d-epsilon-min
                    # In this scenario with one state for each action with p > 1 - eps, 
                    #   a state-state edge can represent an action
                    pi_eps_go[t][p] = pi_eps_go_state

        self.pruned_time_actions = actions
        self.pi_eps_go = pi_eps_go


    def take_action(self, s1, s2, uncertainty):
        # Action is being defined as a state-state transition. 
        #   Possibly use discrete actions in the future in the case that multiple states can
        #   result from a single action with significant probability.

        if np.random.uniform() > uncertainty:
            return s2
        else:
            # Choose next state from possible low probability states
            low_prob_states = self.get_low_prob_neighbors(s1,s2)
            if low_prob_states == []:
                # if no low prob states exist, action must be stay and s2 is the only option
                return s2

            # Choose from low probability states
            s2_new = random.choice(low_prob_states)
            return s2_new

    def get_low_prob_neighbors(self, s1, s2):
        # TODO: put some of this in an mdp class
        # region_to_xy = self.aug_mdp.sig_dict
        region_to_xy = {r:(d['x'],d['y']) for r,d in self.aug_mdp.sig_dict.iteritems()}
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
            z = self.init.keys()[0]
        else:
            z = init_pa_state
            if z not in self.get_states():
                raise Exception("invalid pa state: {}".format(z))
        tau = self.aug_mdp.get_tau()
        init_traj = [z]

        for _ in range(1,tau):
            mdp_s = self.get_mdp_state(z)
            neighbors = self.g.neighbors(z)
            # choose next z with same mdp state
            z = next(iter(filter(lambda next_z: self.get_mdp_state(next_z) == mdp_s, neighbors)))
            init_traj.append(z)
        
        t_init = tau-1
        return z, t_init, init_traj

    def reward(self, pa_s, beta = 2):
        aug_mdp_s = pa_s[0]
        if type(aug_mdp_s) != tuple:
            raise Exception("Invalid augmented mdp state!")
        return self.aug_mdp.reward(aug_mdp_s, beta)

    def new_ep_state(self, last_pa_state):
        # reset DFA state
        aug_mdp_s = last_pa_state[0]
        aug_mdp_init_s = self.aug_mdp.new_ep_state(aug_mdp_s)
        dfa_init_s = self.dfa.init.keys()[0]
        new_pa_s = (aug_mdp_init_s, dfa_init_s)
        if new_pa_s not in self.get_states():
            # This is the case if using the label of s' in the DFA update rather than the label of s.
            # MDP states with a label corresponding to the first hold will not have a PA state with the inital DFA state.
            new_pa_s = (aug_mdp_init_s, dfa_init_s + 1)
        z, _, init_traj = self.initial_state_and_time(new_pa_s)

        return z, init_traj

    def is_accepting_state(self, pa_s):
        dfa_state = self.get_dfa_state(pa_s)
        return dfa_state in self.dfa.final

    def sat(self, pa_s):
        aug_mdp_s = self.get_aug_mdp_state(pa_s)
        return self.aug_mdp.sat(aug_mdp_s)