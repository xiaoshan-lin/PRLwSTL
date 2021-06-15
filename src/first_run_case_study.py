'''
.. module:: Safe RL
	Including Time-left as a state
.. moduleauthor:: Ahmet Semi ASARKAYA <asark001@umn.edu.edu>

'''

import logging, sys
import StringIO
import pdb, os, copy, math
import time, timeit
import operator
import csv

import xlwt 
from xlwt import Workbook 

import os
import scipy as sp # get adjacency
import matplotlib.pyplot
import matplotlib.pyplot as plt
from matplotlib import colors

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

import twtl
import write_files

from collections import Counter
from create_environment import create_ts, update_obs_mat, update_adj_mat_3D,\
								create_input_file, update_adj_mat_3D

from dfa import DFAType
from synthesis import expand_duration_ts, compute_control_policy, ts_times_fsa,\
                      verify, compute_energy                  
from geometric_funcs import check_intersect, downwash_check
from write_files import write_to_land_file, write_to_csv_iter, write_to_csv,\
                        write_to_iter_file, write_to_control_policy_file
from learning import learn_deadlines
from lomap import Ts

from tmdp_stl import tmdp_stl

import IPython

def tau_mdp(ts, ts_weighted, tau):

	def next_mdp_states(state):
		return ts.g.edge[state].keys()
		# all_states = adj_mat[state]
		# adj_states = []
		# for state,cost in enumerate(all_states):
		# 	if cost > 0:
		# 		adj_states.append(state)
		# return adj_states

	def build_states(past, tau):
		if tau == 1:
			# One tau-MDP state per MDP state
			return [tuple(past)]
		next_states = next_mdp_states(past[-1])
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
			more_tmdp_states.extend(build_states(x,tau))
		
		return more_tmdp_states
	
	# make list of tau mdp states where each state is represented by a tuple of mdp states
	tmdp_states = []
	for s in ts.g.nodes():
		tmdp_states.extend(build_states([s],tau))

	# try and recreate process used in ts.read_from_file() except with tau mdp
	# There seems to be a ts with only weights 1 and ts_dict with original weights
	# Looks like it will be easiest to create another nx for weights rather than recreate desired output format
	tmdp = Ts(directed=True, multi=False)
	tmdp_weighted = Ts(directed=True, multi=False)
	tmdp.name = tmdp_weighted.name = "Tau MDP"
	tmdp.init = tmdp_weighted.init = {('Base1',) * tau:1}

	# create dict of dicts representing edges and attributes of each edge to construct the nx graph from
	# attributes are based on the mdp edge between the last (current) states in the tau mdp sequence
	edge_dict = {}
	edge_dict_weighted = {}
	for x1 in tmdp_states:
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
			edge_dict[x1] = {(x2,):edge_attrs[x2] for x2 in next_mdp_states(*x1)}
			edge_dict_weighted[x1] = {(x2,):edge_attrs_weighted[x2][0] for x2 in next_mdp_states(*x1)}

	tmdp.g = nx.from_dict_of_dicts(edge_dict, create_using=nx.MultiDiGraph()) 
	tmdp_weighted.g = nx.from_dict_of_dicts(edge_dict_weighted, create_using=nx.MultiDiGraph()) 

	# add node attributes based on last state in sequence
	for n in tmdp.g.nodes():
		tmdp.g.node[n] = ts.g.node[n[-1]]
	for n in tmdp_weighted.g.nodes():
		tmdp_weighted.g.node[n] = ts_weighted.g.node[n[-1]]

	return(tmdp, tmdp_weighted)
	

def prep_for_learning(ep_len, m, n, h, init_states, obstacles, pick_up_state, delivery_state, rewards, rew_val, custom_flag, custom_task, rewards2, rew_val2, tau):
	# Create the environment and get the TS #
	ts_start_time = timeit.default_timer()
	disc = 1
	TS, obs_mat, state_mat = create_ts(m,n,h)	
	path = '../data/ts_' + str(m) + 'x' + str(n) + 'x' + str(h) + '_1Ag_1.txt'
	paths = [path]
	bases = {init_states[0]: 'Base1'}
	obs_mat = update_obs_mat(obs_mat, state_mat, m, obstacles, init_states[0])
	TS      = update_adj_mat_3D(m, n, h, TS, obs_mat)
	create_input_file(TS, state_mat, obs_mat, paths[0], bases, disc, m, n, h, 0)
	ts_file = paths
	ts_dict = Ts(directed=True, multi=False) 
	ts_dict.read_from_file(ts_file[0])
	ts = expand_duration_ts(ts_dict)
	# make coordinate dict for tmdp_stl
	# apply offset so pos is middle of state
	state_to_pos = dict()
	for s in ts.g.nodes():
		if s == 'Base1':
			state_to_pos[s] = tuple([c + 0.5 for c in init_states[0]])
		else:
			num = int(s[1:])
			#TODO: 3rd dim?
			state_to_pos[s] = ((num // n) + 0.5, (num % n) + 0.5, 0.5)
	stl.set_ts_sig_dict(state_to_pos)
	
	ts_timecost =  timeit.default_timer() - ts_start_time
	# nx.drawing.nx_agraph.view_pygraphviz(ts.g)

	# Create a tau MDP by combining "past" TS states
	# Variables to recreate
	# nx.get_edge_attributes(ts_dict.g,'edge_weight')
	# ts
	tmdp, tmdp_weighted = tau_mdp(ts, ts_dict, tau)


	# Get the DFA #
	dfa_start_time = timeit.default_timer()
	pick_ups = pick_up_state[0][0] * n + pick_up_state[0][1]
	deliveries = delivery_state[0][0] * n + delivery_state[0][1]
	pick_up  = str(pick_ups)   # Check later
	delivery = str(deliveries)
	tf  = str((ep_len-1)/2) # time bound
	if custom_flag == 1:
		phi = custom_task
	else:
		phi = '[H^1 r' + pick_up + ']^[0, ' +  tf + '] * [H^1 r' + delivery + ']^[0,' + tf + ']'  # Construc the task according to pickup/delivery )^[0, ' + tf + ']'
	_, dfa_inf, bdd = twtl.translate(phi, kind=DFAType.Infinity, norm=True) # states and sim. time ex. phi = '([H^1 r47]^[0, 30] * [H^1 r31]^[0, 30])^[0, 30]' 
	dfa_timecost =  timeit.default_timer() - dfa_start_time # DFAType.Normal for normal, DFAType.Infinity for relaxed

	# Get the PA #
	pa_start_time = timeit.default_timer()
	alpha = 1
	nom_weight_dict = {}
	weight_dict = {}
	# pa_or = ts_times_fsa(ts, dfa_inf) # Original pa
	pa_or = ts_times_fsa(tmdp, dfa_inf) # Original pa
	# edges_all = nx.get_edge_attributes(ts_dict.g,'edge_weight')
	edges_all = nx.get_edge_attributes(tmdp_weighted.g,'edge_weight')
	max_edge = max(edges_all, key=edges_all.get)
	norm_factor = edges_all[max_edge]
	for pa_edge in pa_or.g.edges():
		edge = (pa_edge[0][0], pa_edge[1][0], 0)
		nom_weight_dict[pa_edge] = edges_all[edge]/norm_factor
	nx.set_edge_attributes(pa_or.g, 'edge_weight', nom_weight_dict)
	nx.set_edge_attributes(pa_or.g, 'weight', 1)
	pa = copy.deepcopy(pa_or)	      # copy the pa
	time_weight = nx.get_edge_attributes(pa.g,'weight')
	edge_weight = nx.get_edge_attributes(pa.g,'edge_weight')
	for pa_edge in pa.g.edges():
		weight_dict[pa_edge] = alpha*time_weight[pa_edge] + (1-alpha)*edge_weight[pa_edge]
	nx.set_edge_attributes(pa.g, 'new_weight', weight_dict)
	pa_timecost =  timeit.default_timer() - pa_start_time

	# Compute the energy of the states #
	energy_time = timeit.default_timer()
	compute_energy(pa)
	energy_dict = nx.get_node_attributes(pa.g,'energy')
	energy_pa    = []
	for ind in range(len(pa.g.nodes())):
		energy_pa.append(pa.g.nodes([0])[ind][1].values()[0])

	# projection of pa on ts #
	init_state = [init_states[0][0] * n + init_states[0][1]]
	pa2ts = []
	for i in range(len(pa.g.nodes())):
		if pa.g.nodes()[i][0][-1] != 'Base1':
			pa2ts.append(int(pa.g.nodes()[i][0][-1].replace("r","")))
		else:
			pa2ts.append(init_state[0])
			if pa.g.nodes()[i][0] == ('Base1',) * tau:
				i_s = i # Agent's initial location in pa

	# project pa on tmdp
	pa2tmdp = np.zeros(len(pa.g.nodes()))
	for i,x in enumerate(pa.g.nodes()):
		# if n[0] not in tmdp.init.keys():
		pa2tmdp[i] = int(tmdp.g.nodes().index(x[0]))
		if x[0] in tmdp.init.keys():
			i_tmdp = i

	energy_timecost =  timeit.default_timer() - pa_start_time

	# This seems to be unused.
	# TS adjacency matrix and source-target
	# TS_adj = TS
	# TS_s   = []
	# TS_t   = []
	# for i in range(len(TS_adj)):
	# 	for j in range(len(TS_adj)):
	# 		if TS_adj[i,j] != 0:
	# 			TS_s.append(i)
	# 			TS_t.append(j)

	# pa adjacency matrix and source-target 
	pa_adj_st = nx.adjacency_matrix(pa.g)
	pa_adj    = pa_adj_st.todense()
	pa_s = [] # source node
	pa_t = [] # target node
	for i in range(len(pa_adj)):
		for j in range(len(pa_adj)):
			if pa_adj[i,j] == 1:
				pa_s.append(i)
				pa_t.append(j)

    # PA rewards matrix
	# rewards_ts = np.zeros(m * n)#-0.25#
	# rewards_pa = np.zeros(len(pa2ts))
	# rewards_ts_indexes = []
	# for i in range(len(rewards)):
	# 	rewards_ts_indexes.append(rewards[i][0] * n + rewards[i][1]) # rewards_ts_indexes[i] = rewards[i][0] * n + rewards[i][1]		
	# 	rewards_ts[rewards_ts_indexes[i]] = rew_val
	# rewards_ts = np.zeros(m * n)
	# rewards_tmdp = np.zeros(len(tmdp.g.nodes()))
	# rewards_pa = np.zeros(len(pa2ts))
	# rewards_ts_indexes = []
	# j = 0
	# for i in range(len(rewards)+len(rewards2)):
	# 	if i < len(rewards):
	# 		rewards_ts_indexes.append(rewards[i][0] * n + rewards[i][1]) # rewards_ts_indexes[i] = rewards[i][0] * n + rewards[i][1]		
	# 		rewards_ts[rewards_ts_indexes[i]] = rew_val
	# 	else:
	# 		rewards_ts_indexes.append(rewards2[j][0] * n + rewards2[j][1]) # for rew2	
	# 		rewards_ts[rewards_ts_indexes[i]] = rew_val2
	# 		j = j + 1 
	# #print(rewards_ts)
	# for i,x in enumerate(tmdp.g.nodes()):
	# 	sum_reward = 0
	# 	for s in x:
	# 		if s[0] == 'r':
	# 			ts_state = int(s[-1])
	# 			sum_reward += rewards_ts[ts_state]
	# 	rewards_tmdp[i] = sum_reward

	# for i in range(len(rewards_pa)):
	# 	# rewards_pa[i] = rewards_ts[pa2ts[i]]
	# 	rewards_pa[i] = rewards_tmdp[int(pa2tmdp[i])]

	# PA rewards should be assigned according to the Tau MDP projection.
	# Tau MDP awards should be assigned as the smooth max of of the maximum robustness degree according to a given STL
	
	
	# # Display some important info
	print('##### PICK-UP and DELIVERY MISSION #####' + "\n")
	print('Initial Location  : ' + str(init_states[0]) + ' <---> Region ' + str(init_state[0]))
	print('Pick-up Location  : ' + str(pick_up_state[0]) + ' <---> Region ' + pick_up)
	print('Delivery Location : ' + str(delivery_state[0]) + ' <---> Regions ' + delivery)
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

	return i_s, pa, pa_s, pa_t, pa2ts, energy_pa, rewards_pa, pick_up, delivery, pick_ups, deliveries, pa.g.nodes()

def get_possible_actions(pa_g_nodes, energy_pa, pa2ts, pa_s, pa_t, ep_len, Pr_des, eps_unc_learning, pick_up):
	# Remove blocking states and the corresponding transitionswhen the hit flag is raised
	accepting_states = []
	for ind, val in enumerate(energy_pa):   # Find the accepting states
		if val == 0:
			accepting_states.append(ind)

	diff_s2t = []
	nth_row_pa_t = []
	nth_row_pa_s = []
	for i in range(len(pa_s)):
		diff_s2t.append(pa2ts[pa_t[i]] - pa2ts[pa_s[i]])
		nth_row_pa_t.append(int(math.ceil((pa2ts[pa_t[i]]+1.0)/n)))
		nth_row_pa_s.append(int(math.ceil((pa2ts[pa_s[i]]+1.0)/n)))
	
	act_s2t = []
	act_num = [] 			                            #####################################
	for ind, diff in enumerate(diff_s2t):               #                                   #
		if diff == n - 1 and (nth_row_pa_t[ind] != nth_row_pa_s[ind]):                               #Actions and Corresponding Numbering#
			act_s2t.append('SouthWest')					#                                   #
			act_num.append(6)						 	#  --------------          -------  #
		elif diff == n:	                                #  |NW   N    NE|          |0 1 2|  #
			act_s2t.append('South')                     #  |W   Stay  E |  <-----> |3 4 5|  #
			act_num.append(7)                           #  |SW   S    SE|          |6 7 8|  #
		elif diff == n + 1:                             #  --------------          -------  #
			act_s2t.append('SouthEast')                 #                                   #
			act_num.append(8)                           #####################################
		elif diff == -1 and (nth_row_pa_t[ind] == nth_row_pa_s[ind]):
			act_s2t.append('West')
			act_num.append(3)
		elif diff == 0:
			act_s2t.append('Stay')
			act_num.append(4)
		elif diff == 1 and (nth_row_pa_t[ind] == nth_row_pa_s[ind]):
			act_s2t.append('East')
			act_num.append(5)
		elif diff == -n - 1:
			act_s2t.append('NorthWest')
			act_num.append(0)
		elif diff == -n:
			act_s2t.append('North')
			act_num.append(1)
		elif diff == -(n-1) and (nth_row_pa_t[ind] != nth_row_pa_s[ind]):
			act_s2t.append('NorthEast')
			act_num.append(2)
	#print('diff : ', diff_s2t)
	k = 0
	possible_acts = []
	possible_next_states = []
	for i, values in enumerate(Counter(pa_s).values()):
		pos_n_acts = []
		possible_n_next_states = []
		for j in range(values):
			pos_n_acts.append(act_num[k])
			possible_n_next_states.append(pa_t[k])
			k += 1
		possible_acts.append(pos_n_acts)
		possible_next_states.append(possible_n_next_states)

	for ind in sorted(accepting_states, reverse=False):
		possible_next_states.insert(ind, [])
		possible_acts.insert(ind, [])

	#Creating time product MDP
	agent_upt = []
	for i in range(len(pa_g_nodes)):
		if pa_g_nodes[i][1] == 0 or str(pa_g_nodes[i][0][-1]) == 'r'+str(pick_up) : # If the mission changes check here
			agent_upt.append(pa2ts[i])
		else:
			agent_upt.append([])
		
	possible_next_states_time_included_pruned = []
	possible_acts_time_included_pruned = []
	possible_next_states_time_included_not_pruned = []
	possible_acts_time_included_not_pruned = []
	for t_ep in range(ep_len):	
		possible_next_states_copy = copy.deepcopy(possible_next_states)
		possible_acts_copy = copy.deepcopy(possible_acts)
		k_ep = ep_len - t_ep # Remaning episode time   					                                                                                    
		for agent_i in range(len(energy_pa)):
			if energy_pa[agent_i] == 0:
				agent_i = agent_upt.index(pa2ts[agent_i])
			en_list = [energy_pa[i] for i in possible_next_states_copy[agent_i]] # Energies of the next possible states
			not_possible_index = []
			ind_minholder = en_list.index(min(en_list))#np.argmin(np.array(en_list))
			possible_next_states_minholder = possible_next_states_copy[agent_i][ind_minholder] 
			possible_acts_minholder = possible_acts_copy[agent_i][ind_minholder]       
			for j in range(len(possible_next_states_copy[agent_i])):
				d_max   = en_list[j] + 1
				i_max   = int(math.floor((k_ep - 1 - d_max) / 2))
				thr_fun = 0
				for i in range(i_max+1):
					thr_fun = thr_fun + np.math.factorial(k_ep-1) / (np.math.factorial(k_ep-1-i) * np.math.factorial(i)) * eps_unc_learning**i * (1-eps_unc_learning)**(k_ep-i)

				if thr_fun < Pr_des and i_max >= 0: #energy_pa[possible_next_states_copy[agent_s][j]] > k_ep-1: # 
					not_possible_index.append(j)
				
				if i_max < 0:
					not_possible_index.append(j)

			if len(possible_next_states_copy[agent_i]) != 0:		
				for ind in sorted(not_possible_index, reverse=True):
					del possible_next_states_copy[agent_i][ind]
					del possible_acts_copy[agent_i][ind]

			if len(possible_next_states_copy[agent_i]) == 0: # not possible_next_states_copy[agent_s]: #
					possible_next_states_copy[agent_i].append(possible_next_states_minholder)
					possible_acts_copy[agent_i].append(possible_acts_minholder)	
		
		possible_next_states_time_included_pruned.append(possible_next_states_copy)
		possible_next_states_time_included_not_pruned.append(possible_next_states)
		possible_acts_time_included_pruned.append(possible_acts_copy)
		possible_acts_time_included_not_pruned.append(possible_acts)

	return possible_acts_time_included_not_pruned, possible_acts_time_included_pruned, possible_next_states_time_included_not_pruned, possible_next_states_time_included_pruned, act_num


def action_uncertainity(current_act, pa_s, pa_t, act_num, agent_s):	
	indices = []
	acts = []
	for ind, val in enumerate(pa_s):
		if agent_s == val:
			indices.append(ind)
			acts.append(act_num[ind])
	possible_acts = []
	if current_act == 0: # 1,3,4		
		if 1 in acts:
			possible_acts.append(1)
		if 3 in acts:
			possible_acts.append(3)
	elif current_act == 1: # 0,2,4
		if 0 in acts:
			possible_acts.append(0)
		if 2 in acts:
			possible_acts.append(2)
	elif current_act == 2: # 1,5,4
		if 1 in acts:
			possible_acts.append(1)
		if 5 in acts:
			possible_acts.append(5)			
	elif current_act == 3: # 0,6,4
		if 0 in acts:
			possible_acts.append(0)
		if 6 in acts:
			possible_acts.append(6)
	elif current_act == 5: # 2,8,4
		if 2 in acts:
			possible_acts.append(2)
		if 8 in acts:
			possible_acts.append(8)
	elif current_act == 6: # 3,7,4
		if 3 in acts:
			possible_acts.append(3)
		if 7 in acts:
			possible_acts.append(7)
	elif current_act == 7: # 6,8,4
		if 6 in acts:
			possible_acts.append(6)
		if 8 in acts:
			possible_acts.append(8)
	else:                                      # 5,7,4
		if 5 in acts:
			possible_acts.append(5)
		if 7 in acts:
			possible_acts.append(7)

	possible_acts.append(4)	
	chosen_act = random.choice(possible_acts)
	if chosen_act == 4:
		next_state = agent_s
	else:
		inter_ind = acts.index(chosen_act)
		next_state = pa_t[indices[inter_ind]]	

	return chosen_act, next_state


def Q_Learning(Pr_des, eps_unc, eps_unc_learning, N_EPISODES, SHOW_EVERY, LEARN_RATE, DISCOUNT, EPS_DECAY, epsilon, i_s, pa, energy_pa, pa2ts, pa_s, pa_t, act_num, possible_acts_not_pruned, possible_acts_pruned, possible_next_states_not_pruned,possible_next_states_pruned, pick_up, delivery,  pick_ups, deliveries, test_n, n_samples, ts_size):
	
	wb = Workbook()
	sheet_name = 'Simulation' + str(test_n+1) 
	s1 = wb.add_sheet(sheet_name)

	s1.write(1,0,'Task-1')
	s1.write(1+N_EPISODES/SHOW_EVERY,0,'Task-2')
	s1.write(1+2*N_EPISODES/SHOW_EVERY,0,'Task-3')
	s1.write(1+3*N_EPISODES/SHOW_EVERY,0,'Task-4')

	s1.write(0,1,'# of Hit')
	s1.write(0,2,' Avg. Reward')
	s1.write(0,3,' Discounted Avg. Reward')

	s1.write(0,11,' Discounted Episode Reward - Task 1')
	s1.write(0,12,' Discounted Episode Reward - Task 2')
	s1.write(0,13,' Discounted Episode Reward - Task 3')
	s1.write(0,14,' Discounted Episode Reward - Task 4')

	s1.write(0,6,'Total Run Time')
	s1.write(0,7,'Total Avg. Reward')

	inx = 0


	QL_start_time = timeit.default_timer()

	EVERY_PATH = []
	episode_rewards = []

	# Initialize the Q - table (Between -0.01 and 0)
	pa_size = []
	q_table = []	
	agent_s = []
	hit_count = []
	mission_tracker = []
	ep_per_task = []
	disc_ep_per_task = []
	old_q_tables = []
	all_samples = []
	for i in range(len(energy_pa)):		
		pa_size.append(len(pa[i].g.nodes()))
		agent_s.append(i_s[i])      # Initialize the agent's location
		hit_count.append(0)
		mission_tracker.append(0)
		ep_per_task.append([])
		disc_ep_per_task.append([])
		all_samples.append([])
		q_table.append([])
		old_q_tables.append([])
		for t in range(ep_len+1):
			q_table[i].append(np.random.rand(pa_size[i],9) * 0.001 - 0.001)  # of states x # of actions
			old_q_tables[i].append(q_table[i][t])

	ep_rewards = [] 
	ep_trajectories_pa = []

	agent_upt_i = []
	agent_upt = []
	for j in range(len(energy_pa)):
		for i in range(len(pa[j].g.nodes())):
			if pa[j].g.nodes()[i][1] == 0 or str(pa[j].g.nodes()[i][0][-1]) == 'r'+str(pick_up[j]) :#or str(pa[j].g.nodes()[i][0]) == 'r'+str(delivery[j]): # If the mission changes check here
				agent_upt_i.append(pa2ts[j][i])
			else:
				agent_upt_i.append([])
		agent_upt.append(agent_upt_i)

	for episode in range(N_EPISODES):
		# if episode > 900000: # can be switch to only exploitation after some episode
		# 	epsilon = 0

		which_pd = np.random.randint(len(energy_pa)) # randomly chosing the pick_up delivery states

		mission_tracker[which_pd] = mission_tracker[which_pd] + 1
		hit = []
		ep_rew = []
		for i in range(len(energy_pa)):
			hit.append(0)

		#TODO reevaluate if this works
		# Reset agent_s
		agent_s = [i_s[which_pd]]

		ep_traj_pa = [agent_s[which_pd]] # Initialize the episode trajectory
		ep_rew     = 0         # Initialize the total episode reward
		disc_ep_rew = 0

		for t_ep in range(ep_len):

			old_q_tables[which_pd][t_ep] = q_table[which_pd][t_ep]

			possible_acts = possible_acts_not_pruned[which_pd]
			possible_next_states = possible_next_states_not_pruned[which_pd]

			if hit[which_pd] == 0:      					                                                                                    
				if energy_pa[which_pd][agent_s[which_pd]] == 0:  # Raise the 'hit flag' if the mission is achieved 
					hit[which_pd] = 1                  # 	
					agent_s[which_pd] = agent_upt[which_pd].index(pa2ts[which_pd][agent_s[which_pd]]) # 	
					hit_count[which_pd] = hit_count[which_pd] + 1
				else:
					possible_acts = possible_acts_pruned[which_pd]
					possible_next_states = possible_next_states_pruned[which_pd]

			if len(possible_acts[t_ep][agent_s[which_pd]]) == 0:
				agent_s[which_pd] = agent_upt[which_pd].index(pa2ts[which_pd][agent_s[which_pd]])
					
			if np.random.uniform() > epsilon:                              # Exploit
				possible_qs = q_table[which_pd][t_ep][agent_s[which_pd], possible_acts[t_ep][agent_s[which_pd]]] # Possible Q values for each action
				next_ind    = np.argmax(possible_qs)                       # Pick the action with max Q value 
			else:                                                          # Explore
				next_ind  = np.random.randint(len(possible_acts[t_ep][agent_s[which_pd]])) # Picking a random action
			# Taking the action
			prev_state = agent_s[which_pd]
			intended_action = possible_acts[t_ep][prev_state][next_ind]
			if np.random.uniform() < eps_unc:
				[chosen_act, next_state] = action_uncertainity(intended_action, pa_s[which_pd], pa_t[which_pd], act_num[which_pd], agent_s[which_pd])
				action    = chosen_act
				s_a       = (agent_s[which_pd], action)                                   # State & Action pair
				agent_s[which_pd]   = next_state # possible_next_states
			else:
				action    = intended_action
				s_a       = (agent_s[which_pd], action)                                   # State & Action pair
				agent_s[which_pd]   = possible_next_states[t_ep][agent_s[which_pd]][next_ind]        # moving to next state  (s,a)

			ep_traj_pa.append(agent_s[which_pd])			
			current_q = q_table[which_pd][t_ep][prev_state, intended_action]
			max_future_q = np.amax(q_table[which_pd][t_ep+1][agent_s[which_pd], :])                                          # Find the max future q 	
			# rew_obs      = rewards_pa[which_pd][agent_s[which_pd]] * np.random.binomial(1, 1-rew_uncertainity)   # Observe the rewards of the next state
			# new_q        = (1 - LEARN_RATE) * current_q + LEARN_RATE * (rew_obs + DISCOUNT * max_future_q)

			# rew_obs should be exp( sign * beta * internal robustness degree of "next" tau-MDP state)
			beta = 50
			next_tmdp_s = pa[which_pd].g.nodes()[agent_s[which_pd]][0]
			sign,rd_phi = stl.rdegree_lse(next_tmdp_s)

			# # temp
			# foo1,foo2 = stl.rdegree_lse(('r3','r6','r9'), 2)

			# sum from t=tau-1 to end
			if t_ep < stl.get_tau() - 1:
				rew_obs = 0
			else:
				rew_obs = sign * np.exp(sign * beta * rd_phi) * np.random.binomial(1, 1-rew_uncertainity)
			# if rd_phi == None:
			# 	# If rd_phi is none, then this time step is not included in the lse sum
			# 	rew_obs = 0
			# else:
			# 	rew_obs = sign * np.exp(sign * beta * rd_phi) * np.random.binomial(1, 1-rew_uncertainity)
			# TODO check purpose of rew_uncertain(i)ty
			new_q = (1 - LEARN_RATE) * current_q + LEARN_RATE * (rew_obs + DISCOUNT * max_future_q)

			q_table[which_pd][t_ep][prev_state, intended_action] = new_q 

			disc_ep_rew += rew_obs * (DISCOUNT ** t_ep)                                                 
			ep_rew += rew_obs

			# Adding sample to the memory
			all_samples[which_pd].append([prev_state, intended_action, rew_obs, agent_s[which_pd], t_ep]) # S,A,R,S' and time
			
			# Sample n times
			for i in range(n_samples):
				random_sample_index = np.random.choice(len(all_samples[which_pd]))
				sample_s       = all_samples[which_pd][random_sample_index][0]
				sample_action  = all_samples[which_pd][random_sample_index][1]
				sample_r       = all_samples[which_pd][random_sample_index][2]
				sample_s_prime = all_samples[which_pd][random_sample_index][3]
				sample_t       = all_samples[which_pd][random_sample_index][4]

				current_q = q_table[which_pd][sample_t][sample_s, sample_action]
				max_future_q = np.amax(q_table[which_pd][sample_t+1][sample_s_prime, :])
				new_q        = (1 - LEARN_RATE) * current_q + LEARN_RATE * (sample_r + DISCOUNT * max_future_q)
				q_table[which_pd][sample_t][sample_s, sample_action] = new_q 

		# complete the lse so the rewards make sense
		ep_rew = sign * (1.0/beta) * np.log(sign * ep_rew)
		agent_s[which_pd] = agent_upt[which_pd].index(pa2ts[which_pd][agent_s[which_pd]]) # Re-initialize after the episode is finished
		ep_rewards.append(ep_rew)
		ep_trajectories_pa.append(ep_traj_pa)
		epsilon = epsilon * EPS_DECAY
		disc_ep_per_task[which_pd].append(disc_ep_rew)
		ep_per_task[which_pd].append(ep_rew)		
		if (episode+1) % SHOW_EVERY == 0:
			inx = inx + 1
			for ind in range(len(energy_pa)):
				avg_per_task = np.mean(ep_per_task[ind])
				disc_avg_per_task = np.mean(disc_ep_per_task[ind])
				print('Episode # ' + str(episode+1) + ' : Task-' + str(ind) + '   # of Hit=' + str(len(ep_per_task[ind])) + '   Avg.=' + str(avg_per_task))
				s1.write(ind*N_EPISODES/SHOW_EVERY+inx,1,len(ep_per_task[ind]))
				s1.write(ind*N_EPISODES/SHOW_EVERY+inx,2,avg_per_task)
				s1.write(ind*N_EPISODES/SHOW_EVERY+inx,3,disc_avg_per_task)

		if (episode+1) % SHOW_EVERY == 0:
			avg_rewards = np.mean(ep_rewards[episode-SHOW_EVERY +1: episode])
			print('Episode # ' + str(episode+1) + ' : Epsilon=' + str(round(epsilon, 4)) + '    Avg. reward in the last ' + str(SHOW_EVERY) + ' episodes=' + str(round(avg_rewards,2)))
	
	best_episode_index = ep_rewards.index(max(ep_rewards))
	optimal_policy_pa  = ep_trajectories_pa[N_EPISODES-1]#ep_trajectories_pa[best_episode_index] # Optimal policy in pa  ep_trajectories_pa[N_EPISODES-1]#
	optimal_policy_ts  = []                                     # optimal policy in ts
	opt_pol            = []                                     # optimal policy in (m, n, h) format for visualization
	for ind, val in enumerate(optimal_policy_pa):
		optimal_policy_ts.append(pa2ts[which_pd][val])
		opt_pol.append((math.floor(optimal_policy_ts[ind]/n), optimal_policy_ts[ind]%n, 0))
	
	print('Tajectory at the last episode : ' + str(optimal_policy_ts))


	indices=[0,1,2]#, 50000,50001,50001,100000,100001,100002,299997,299998,299999,N_EPISODES-3,N_EPISODES-2,N_EPISODES-1
	optimal_policy_pas = []
	for i in range(len(indices)):		
		optimal_policy_pas.append(ep_trajectories_pa[indices[i]])
		optimal_policy_ts  = []
		for ind, val in enumerate(optimal_policy_pas[i]):
			optimal_policy_ts.append(pa2ts[which_pd][val])
		#print('Tajectory at the episode ' +  str(indices[i]) + ' : '+str(optimal_policy_ts))

	QL_timecost =  timeit.default_timer() - QL_start_time
	success_ratio = []
	for i in range(len(energy_pa)):
		success_ratio.append(100*hit_count[i]/mission_tracker[i])
		print("Successful Mission Ratio[%] = " + str(success_ratio[i]))
		print("Successful Missions = " + str(hit_count[i]) + " out of " + str(mission_tracker[i]))
	d_maxs = []
	for i in range(len(energy_pa)):
		d_maxs.append(max(energy_pa[i]))
	max_energy   = max(d_maxs)

	# for i in range(len(energy_pa)):
	# 	for j in range(ep_len):
	# 		#name_diff = "q_table_diff_perc_" + str(i) + ".npy"
	# 		#name_q    = "Env3_Converged_Q_TABLE_GNC" + str(n_samples) + '_task'+ str(i) + '_t' + str(j) + ".npy"
	# 		#np.save(name_diff,q_table_diff_perc)
	# 		#np.save(os.path.join('Q_TABLES',name_q) ,q_table[i][j])
	
	print('Total time for Q-Learning : ' + str(QL_timecost) + ' seconds')
	print('Action uncertainity[%] = ' + str(eps_unc*100))
	print('# of Samples = ' + str(n_samples))
	print("Desired Minimum Success Ratio[%] = " + str(100*Pr_des))
	print("Episode Length = " + str(ep_len) + "  and  Max. Energy of the System = " + str(max_energy))    
	print('Reward at last episode = '+str(ep_rewards[-1]))

	# for task in range(len(energy_pa)):
	# 	for ind in range(len(disc_ep_per_task[task])):
	# 		s1.write(1+ind,11+task,disc_ep_per_task[task][ind])

	s1.write(1,6,QL_timecost)
	s1.write(1,7,np.mean(ep_rewards))
	filename = 'hybrid_n' +str(n_samples) +'.xls'
	filename = filename
	wb.save(filename) 

	return opt_pol


def visualization(m, n, init_states, obstacles, pick_up_state, delivery_state, rewards, opt_pol):
	# Trajectory
	x_data = np.zeros(len(opt_pol))
	y_data = np.zeros(len(opt_pol))
	for i in range(len(opt_pol)):
	    x_data[i] = opt_pol[i][1]
	    y_data[i] = opt_pol[i][0]

	# Color accordingly
	cdata = np.zeros((m, n))
	for i in range(len(obstacles)):
	   cdata[obstacles[i][0], obstacles[i][1]] = 0.75

	for i in range(len(rewards)):
	    cdata[rewards[i][0], rewards[i][1]] = 2.75

	cdata[init_states[0][0], init_states[0][1]] = 1.25
	cdata[pick_up_state[0][0], pick_up_state[0][1]] = 1.75
	cdata[delivery_state[0][0], delivery_state[0][1]] = 2.25
	cdata[rewards[0][0], rewards[0][1]] = 2.75

	cmap = colors.ListedColormap(['white', 'red', 'yellow', 'blue', 'green', 'gray'])
	bounds = [0, 0.5 ,1, 1.5, 2, 2.5, 3]
	norm = colors.BoundaryNorm(bounds, cmap.N)

	# Create figure
	fig, ts = plt.subplots()    
	fig.suptitle('Transition System')
	ts.imshow(cdata, cmap=cmap, norm=norm)

	ts.plot(x_data,y_data, linewidth=6)

	plt.tick_params(
	axis='both',       # changes apply to both axes
	which='major',     # major ticks are affected
	bottom=False,      # ticks along the bottom edge are off
	left =False,       # ticks along the left edge are off
	labelbottom=False, # labels along the bottom edge are off
	labelleft  =False) # labels along the left edge are off

	# draw gridlines
	ts.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)    
	ts.set_xticks(np.arange(0.5, n, 1));
	ts.set_yticks(np.arange(0.5, m, 1));
	plt.show()

### Main Code Here ###
if __name__ == '__main__':
	os.system('clear')
	start_time  = time.time()
	custom_flag = 0          # Task flag, if zero, it is a pick-up delivery mission. If 1, define a new custom task
	custom_task = '[H^1 r46]^[0,10] * ([H^1 r57]^[0, 10] | [H^1 r24]^[0, 10])  * [H^1 Base1]^[0,10]' # '[H^1 r46]^[0,10] * ([H^1 r57]^[0, 10] | [H^1 r24]^[0, 10])  * [H^1 Base1]^[0,10]'
	##### System Inputs for Data Prep. #####
	# ep_len = 21 # Episode length
	m = 4       # of rows
	n = 4       # of columns  8
	h = 1       # height set to 1 for 2D case
	ts_size =m*n

	# STL constraint
	stl_expr = 'G[0,20]F[0,2]((x>2)&(x<3)&(y>2)&(y<3))'
	# stl_expr = 'F[0,20]G[0,2]((x>2)&(x<3)&(y>2)&(y<3))'
	stl = tmdp_stl(stl_expr)
	tau = stl.get_tau()
	ep_len = int(stl.hrz())
	
	# Specify initial states and obstacles (row,column,altitude/height)
	init_states    = [(0,0,0)]                                         # Specify initial states and obstacles (row,column,altitude/height)
	# obstacles = [(2,3,0), (2,4,0), (3,3,0), (3,4,0)]
	obstacles = []
	

	# Specify pick-up and delivery locations
	pick_up_state = []
	delivery_state = []                                      
	pick_up_state.append([(3,3,0)])
	# pick_up_state.append([(6,6,0)])                                         
	delivery_state.append([(1,3,0)])                                         
	# delivery_state.append([(5,3,0)])                                         

	# Specify the reward locations, and reward uncertainty
	# rewards   = [(4,0,0), (4,1,0), (5,0,0), (5,1,0)] # Reward 1 Reward Locations
	rewards = [(1,2,0)]
	rewards2  = []                  # Reward 2 Rewards Locations
	rew_val   = 1  					# Reward 1 value
	rew_val2  = 2  					# Reward 2 value
	rew_uncertainity = 0.00 		# Reward Uncertainty

	##### System Inputs for Q-Learning #### 	#For debugging
	LEARN_FLAG = True  # False # If true learn a new Q table, if False load the previously found one
	sample_size = 10000 # Specify How Many samples to run

	N_EPISODES = 400000      # of episodes
	SHOW_EVERY = 5000       # Print out the info at every ... episode
	# LEARN_RATE = 0.1
	LEARN_RATE = 0.9
	DISCOUNT   = 0.95
	EPS_DECAY  = 0.999985 #0.99989
	epsilon    = 0.4# exploration trade-off
	eps_unc    = 0.03 # Uncertainity in actions, real uncertainnity in MDP
	eps_unc_learning = 0.05 # Overestimated uncertainity used in learning
	Pr_des     = 0.85 # Minimum desired probability of satisfaction

	n_samples_all = [50] # Running the algorithm for different model based samples, 0 for model free learning

	# Call the function 'prep_for_learning' and 'get_possible_actions' to get required parameters for learning #
	prep_start_time = timeit.default_timer()
	energy_pa = []
	pa = []
	pa_s = []
	pa_t = []
	pa2ts = []
	pick_up = []
	delivery = []
	possible_acts_pruned = []
	possible_acts_not_pruned = []
	possible_next_states_pruned = []
	possible_next_states_not_pruned = []
	act_num = []
	i_s = []
	rewards_pa = []
	pick_ups = [] 
	deliveries = []
	for ind_p in range(len(pick_up_state)):
		for ind in range(len(delivery_state)):
			[i_s_i, pa_i, pa_s_i, pa_t_i, pa2ts_i, energy_pa_i, rewards_pa_i, pick_up_i, delivery_i,  pick_ups_i, deliveries_i, pa_g_nodes_i] = prep_for_learning(ep_len, m, n, h, init_states, obstacles, pick_up_state[ind_p], delivery_state[ind], rewards, rew_val, custom_flag, custom_task, rewards2, rew_val2, tau)
			i_s.append(i_s_i)
			rewards_pa.append(rewards_pa_i)
			pa.append(pa_i)
			pa_s.append(pa_s_i)
			pa_t.append(pa_t_i)
			pa2ts.append(pa2ts_i)
			energy_pa.append(energy_pa_i)
			pick_up.append(pick_up_i)
			delivery.append(delivery_i)
			pick_ups.append(pick_ups_i)
			deliveries.append(deliveries_i)
			[possible_acts_time_included_not_pruned_i, possible_acts_time_included_pruned_i, possible_next_states_time_included_not_pruned_i, possible_next_states_time_included_pruned_i, act_num_i] = get_possible_actions(pa_g_nodes_i,energy_pa_i, pa2ts_i, pa_s_i, pa_t_i, ep_len, Pr_des, eps_unc, pick_up_i)
			possible_acts_not_pruned.append(possible_acts_time_included_not_pruned_i)
			possible_acts_pruned.append(possible_acts_time_included_pruned_i)
			possible_next_states_not_pruned.append(possible_next_states_time_included_not_pruned_i)
			possible_next_states_pruned.append(possible_next_states_time_included_pruned_i)
			act_num.append(act_num_i)

	prep_timecost =  timeit.default_timer() - prep_start_time
	print('Total time for data prep. : ' + str(prep_timecost) + ' seconds \n')


	if LEARN_FLAG:
		# Check possible minimum threshold
		d_maxs = []
		for i in range(len(energy_pa)):	
			d_maxs.append(max(energy_pa[i]))
		d_max   = max(d_maxs)
		i_max   = int(math.floor((ep_len - 1 - d_max) / 2))
		thr_fun = 0
		for i in range(i_max+1):
			thr_fun = thr_fun + np.math.factorial(ep_len-1) / (np.math.factorial(ep_len-1-i) * np.math.factorial(i)) * eps_unc_learning**i * (1-eps_unc_learning)**(ep_len-i)
		print('Maximum threshold that can be put ' + str(thr_fun))	
		if thr_fun < Pr_des:
			print('Please set a less desired probability threshold than ' + str(thr_fun))
		else:
			# Call the Q_Learning Function
			for n_samples in n_samples_all: 
				test_n = 0
				opt_pol = Q_Learning(Pr_des, eps_unc, eps_unc_learning, N_EPISODES, SHOW_EVERY, LEARN_RATE, DISCOUNT, EPS_DECAY, epsilon, i_s, pa, energy_pa, pa2ts, pa_s, pa_t, act_num, possible_acts_not_pruned, possible_acts_pruned, possible_next_states_not_pruned,possible_next_states_pruned, pick_up, delivery,  pick_ups, deliveries, test_n, n_samples, ts_size)
				print("Total run time : " + str((time.time() - start_time))+' seconds \n \n \n')

			##### Visualize ######
			#visualization(m, n, init_states, obstacles, pick_up_state[0], delivery_state[0], rewards, opt_pol)
	else:
		Q_TABLEs = []
		agent_upt = []
		n_samples = 10 # choose for which n_samples to run simulation
		for ind in range(len(energy_pa)):
			Q_TABLEs.append([])
			for j in range(ep_len):
				name_q    = "Env3_Converged_Q_TABLE_GNC" + str(n_samples) + '_task'+ str(ind) + '_t' + str(j) + ".npy" # name should be the same as defined in Q learning function
				Q_TABLEs[ind].append(np.load(os.path.join('Q_TABLES',name_q), allow_pickle=True))
			avg_tot_und_rew = 0
			avg_tot_disc_rew = 0
			#
			agent_upt_i = []
			for i in range(len(pa[ind].g.nodes())):
				if pa[ind].g.nodes()[i][1] == 0 or str(pa[ind].g.nodes()[i][0]) == 'r'+str(pick_up[ind]):
					agent_upt_i.append(pa2ts[ind][i])
				else:
					agent_upt_i.append([])
				if pa[ind].g.nodes()[i][1] == 0 and pa[ind].g.nodes()[i][0] == 'Base1':
					starting_state = i
			agent_upt.append(agent_upt_i)
			#
			hit_raise = 0
			for k in range(sample_size):
				hit_flag = False
				agent_s = starting_state
				tot_und_rew = 0
				tot_disc_rew = 0
				for t_ep in range(ep_len):
					possible_acts = possible_acts_not_pruned[ind]
					possible_next_states = possible_next_states_not_pruned[ind]

					if hit_flag == False:
						if energy_pa[ind][agent_s] == 0:
							hit_raise += 1
							hit_flag = True
							agent_s = agent_upt[ind].index(pa2ts[ind][agent_s])
						else:
							possible_acts = possible_acts_pruned[ind]
							possible_next_states = possible_next_states_pruned[ind]

					prev_state = agent_s
					possible_qs = Q_TABLEs[ind][t_ep][prev_state, possible_acts[t_ep][prev_state]] # Possible Q values for each action
					next_ind    = np.argmax(possible_qs)
					intended_action = possible_acts[t_ep][prev_state][next_ind]

					if np.random.uniform() < eps_unc - 0.07:
						[chosen_act, agent_s] = action_uncertainity(intended_action, pa_s[ind], pa_t[ind], act_num[ind], prev_state)
					else:
						agent_s = possible_next_states[t_ep][prev_state][next_ind]

					rew_obs = rewards_pa[ind][agent_s]
					tot_und_rew += rew_obs
					tot_disc_rew += rew_obs * (DISCOUNT ** t_ep)

				avg_tot_und_rew += tot_und_rew
				avg_tot_disc_rew += tot_disc_rew
			success_ratio = float(hit_raise) / float(sample_size) * 100.0
			print('Mission ' + str(ind))
			print('Success Ratio[%] = '+ str(success_ratio))
			print('Total Undiscounted Reward after learned policy = '+str(avg_tot_und_rew / sample_size))
			print('Total Discounted Reward after learned policy = '+str(avg_tot_disc_rew / sample_size))
			print('\n')