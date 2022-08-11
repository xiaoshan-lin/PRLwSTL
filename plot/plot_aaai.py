import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import numpy as np
import os
import yaml

this_file_path = os.path.dirname(os.path.abspath(__file__))

def plot_result(proj_dir):
    alpha = 0.3
    
    print('plot result ...')
    if not proj_dir.strip():
        root = tk.Tk()
        root.withdraw()
        cfg_path_flag = filedialog.askopenfilenames(initialdir=os.path.dirname(os.path.abspath(__file__))+'/../result',
                                               filetypes=[('YAML','*.yaml')])
        cfg_path_tau = filedialog.askopenfilenames(initialdir=os.path.dirname(os.path.abspath(__file__))+'/../result',
                                               filetypes=[('YAML','*.yaml')])
        root.destroy()    
        if cfg_path_flag==() or cfg_path_tau==():
            return False, None
        
        with open(cfg_path_flag[0], 'r') as f:
            config_flag = yaml.safe_load(f)
        proj_dir_flag = os.path.dirname(cfg_path_flag[0])
        with open(cfg_path_tau[0], 'r') as f:
            config_tau = yaml.safe_load(f)
        proj_dir_tau = os.path.dirname(cfg_path_tau[0])

        aaai_dir = os.path.join(this_file_path,'../result/constant_epsln/aaai')
    
    colors = [(174/255,199/255,232/255),(1,187/255,120/255),(152/255,223/255,138/255),(1,152/255,150/255),
              (219/255,219/255,141/255),(199/255,199/255,199/255),(158/255,218/255,229/255),(196/255,156/255,148/255)]
    num_episodes = config_flag['Q-learning config']['number of episodes']
    des_prob = config_flag['TWTL constraint']['desired satisfaction probability']
    npz_list_flag = []
    npz_list_tau = []
    
    for root, dirs, files in os.walk(proj_dir_flag):
          for f in files:
              if f.endswith(".npz"):
                  npz_list_flag.append(np.load(os.path.join(root,f), 'rb',allow_pickle=True))

    for root, dirs, files in os.walk(proj_dir_tau):
          for f in files:
              if f.endswith(".npz"):
                  npz_list_tau.append(np.load(os.path.join(root,f), 'rb',allow_pickle=True))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111) 
    sat_arr_flag = np.array([i['stl_sat_rate'].tolist() for i in npz_list_flag]) 
    sat_arr_tau = np.array([i['stl_sat_rate'].tolist() for i in npz_list_tau]) 
    sat_max_flag = np.amax(sat_arr_flag,0)
    sat_min_flag = np.amin(sat_arr_flag,0)
    sat_mean_flag = np.mean(sat_arr_flag,0)

    sat_max_tau = np.amax(sat_arr_tau,0)
    sat_min_tau = np.amin(sat_arr_tau,0)
    sat_mean_tau = np.mean(sat_arr_tau,0)
    ax.fill_between(np.arange(500,num_episodes+500,500), sat_min_flag, sat_max_flag, 
                     facecolor=colors[2], alpha=alpha)
    a1, = ax.plot(np.arange(500,num_episodes+500,500),sat_mean_flag,c=colors[2],linewidth=3,label='flag-MDP')

    ax.fill_between(np.arange(500,num_episodes+500,500), sat_min_tau, sat_max_tau, 
                     facecolor=colors[0], alpha=alpha)
    a2, = ax.plot(np.arange(500,num_episodes+500,500),sat_mean_tau,c=colors[0],linewidth=3,label=r'$\tau$-MDP')

    '''
    reward_arr = np.array([i['ep_rewards'].tolist() for i in npz_list])         
    reward_max = np.amax(reward_arr,0)
    reward_min = np.amin(reward_arr,0)
    reward_mean = np.mean(reward_arr,0)

    # calculate moving average
    average_max = []
    average_mean = []
    average_min = []
    window = 50
    for i in range(num_episodes):
        right = min(i+int(window/2),num_episodes)
        left = max(0,i-int(window/2))
        average_max.append(sum(reward_max[left:right])/(right-left))
        average_min.append(sum(reward_min[left:right])/(right-left))
        average_mean.append(sum(reward_mean[left:right])/(right-left))    
   
       
    ax.fill_between(range(num_episodes), average_min, average_max, 
                     facecolor=colors[2], alpha=alpha)
    a1, = ax.plot(range(num_episodes),average_mean,c=colors[2],linewidth=3,label='reward')'''

    pp = (a1,a2)
    #ax.set_ylim(-1,50)
    ax.legend(handles=pp,loc='best',frameon=False,ncol=1,prop={'size': 20})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Number of Episodes', fontsize=18)
      
    '''
    ax = fig.add_subplot(224)      
    a1, = ax.plot(range(len(sat_prob_1)),sorted_prob_1,c=colors[6],linewidth=3,label='new lower bound')
    a2, = ax.plot(range(len(sat_prob)),sorted_prob,c=colors[5],linewidth=3,label='lower bound')
    a3, = ax.plot(range(len(sat_prob_1)),[des_prob]*len(sat_prob_1),c='k',linewidth=3,
                   label='desired probability',linestyle='dashed')
    pp = (a1,a2,a3)
    ax.legend(handles=pp,loc='best',frameon=False,ncol=1,prop={'size': 15})'''
    plt.title('STL Satisfaction Rate by Number of Episodes',fontsize=20)
    plt.tight_layout()
   
    print('Saving figure ...')
    plt.savefig(aaai_dir+'/result.png', dpi=600)
    print('Figure Saved')
    plt.show()
    return True, proj_dir_flag

if __name__ == "__main__":
    #plot_test_result('')
    plot_result('')
