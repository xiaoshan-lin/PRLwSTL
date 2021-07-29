
import lomap
import synthesis as synth
import networkx as nx
import math

class AugPa(lomap.Model):

    def __init__(self, aug_mdp, dfa):
        # aug_mdp is an augmented mdp such as a tau-mdp or flag-mdp
        # dfa is generated from a twtl constraint

        lomap.Model.__init__(self, directed=True, multi=False)

        self.aug_mdp = aug_mdp
        self.dfa = dfa

        # generate
        synth.ts_times_fsa(aug_mdp, dfa, self)

        # allow caller to compute energy so it can be timed
        self.energy_dict = None

    def get_mdp_state(self, pa_state):
        aug_mdp_state = pa_state[0]
        return self.aug_mdp.get_mdp_state(aug_mdp_state)

    def compute_energy(self):

        #TODO increase speed significantly by computing energy over pa of simple mdp and dfa
        #   and then projecting to this pa

        synth.compute_energy(self)
        self.energy_dict = nx.get_node_attributes(self.g, 'energy')

    def get_energy(self, pa_state):
        return self.energy_dict[pa_state]

    def prune_actions(self, eps_uncertainty, des_prob):

        # TODO actually use actions and not next state

        if self.energy_dict == None:
            raise Exception("Please call compute_energy before prune_actions")

        ep_len = self.aug_mdp.get_hrz()
        # initialize actions of time product MDP
        actions = [nx.convert.to_dict_of_lists(self.g) for _ in range(ep_len + 1)]

        pi_eps_go = {t:{} for t in range(ep_len)}

        # create set of non-accepting states
        accepting_states = self.final
        non_accepting_states = actions[0].keys()
        for s in accepting_states:
            non_accepting_states.remove(s)
        
        for p in non_accepting_states:
            for t in range(ep_len):
                pi_eps_go_state = None
                dmax_min = float('inf')
                # Create copy to avoid removing from the same list being iterated
                next_ps = actions[t][p][:]
                for next_p in next_ps:
                    dmax = self.energy_dict[next_p]
                    k = ep_len - t
                    imax = int(math.floor((k - 1 - dmax) / 2))

                    # track minimum energy for the case that the action set ends empty
                    if dmax < dmax_min:
                        dmax_min = dmax
                        pi_eps_go_state = next_p

                    sat_prob = 0
                    for i in range(imax + 1):
                        part_1 = math.factorial(k - 1) / (math.factorial(k - 1 - i) * math.factorial(i))
                        part_2 = eps_uncertainty ** i
                        part_3 = (1 - eps_uncertainty) ** (k - 1 - i)
                        sat_prob += part_1 * part_2 * part_3

                    if imax < 0 or sat_prob < des_prob:
                        # prune this action
                        actions[t][p].remove(next_p)
                
                if actions[t][p] == []:
                    # record state signifying action minimizing d-epsilon-min
                    # In this scenario with one state for each action with p > 1 - eps, 
                    #   a state-state edge can represent an action
                    pi_eps_go[t][p] = pi_eps_go_state

        self.pruned_time_actions = actions
        self.pi_eps_go = pi_eps_go
