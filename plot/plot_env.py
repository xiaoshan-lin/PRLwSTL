import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from matplotlib.animation import FuncAnimation
import time
from numpy import remainder as rem
from matplotlib.lines import Line2D
import yaml
import tkinter as tk
from tkinter import filedialog
import os
import re


#TODO This ugly code needs to re-organized
'''
  *  Author: Xiaoshan Lin
  *  This code is used to plot the environment
  *  If plot_traj set to True, testing trajectories save in trajectory_log.txt will be plot
  *  Trajectories will be plotted one by one, once the matplotlib window is launched, click on your mouse 
     to stop and save the plots, click on the mouse again to resume displaying
'''

def plot_env(proj_dir):
    # if proj_dir='' choose yaml file in the popped window
    # else plot the environment acoording to the default.yaml file in proj_dir
    if not proj_dir.strip():
        root = tk.Tk()
        root.withdraw()
        cfg_path = filedialog.askopenfilenames(initialdir="../result",filetypes=[('YAML','*.yaml')])       
        root.destroy()  
        if cfg_path==():
            return False  
        with open(cfg_path[0], 'r') as f:
            config = yaml.safe_load(f)
        proj_dir = os.path.dirname(cfg_path[0])
    else:
        with open(proj_dir+'/default.yaml', 'r') as f:
            config = yaml.safe_load(f)

    height = config['environment']['height']
    width = config['environment']['width']
    obs_list = config['environment']['obstacles']
    prop_dict = config['environment']['custom prop dict']
    test_iters = config['Testing']['test_iters']
    e = 0.05
    color=[(174/255,199/255,232/255),(1,187/255,120/255),(152/255,223/255,138/255),(1,152/255,150/255),(165/255,172/255,175/255),\
           (96/255,99/255,106/255),(214/255,39/255,40/255)]
    label_list = []
    plot_traj = False
  
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    # plot obstacles
    plot_obstacle = False
    
    if obs_list != None and plot_obstacle:
      col_idx = [i[1] for i in obs_list]
      row_idx = [i[0] for i in obs_list]
      x = [i + 0.5 for i in col_idx]
      y = [h - 0.5 - i for i in row_idx]
      obs_xy = [(i,j) for i,j in zip(x,y)] 
      for i in obs_xy:
        plt.plot([[i[0]-0.5+e,i[0]-0.5+e,i[0]-0.5+e,i[0]-0.17+e,i[0]+0.16+e],\
                  [i[0]-0.17-e,i[0]+0.16-e,i[0]+0.5-e,i[0]+0.5-e,i[0]+0.5-e]],\
                  [[i[1]+0.17+e,i[1]-0.16+e,i[1]-0.5+e,i[1]-0.5+e,i[1]-0.5+e],\
                  [i[1]+0.5-e,i[1]+0.5-e,i[1]+0.5-e,i[1]+0.17-e,i[1]-0.16-e]],'k',linewidth=0.5)
        plt.plot([i[0]-0.5,i[0]+0.5,i[0]+0.5,i[0]-0.5,i[0]-0.5],\
                  [i[1]+0.5,i[1]+0.5,i[1]-0.5,i[1]-0.5,i[1]+0.5],'k',linewidth = 0.8)

    # plot regions    
    if config['environment']['custom prop']:
        regions = list(prop_dict.values())
        regions_dict = {r:{} for r in regions}
        regions_idx = list(prop_dict.keys())
        for i in regions_idx:
            if regions_dict[prop_dict[i]]=={}:
                regions_dict[prop_dict[i]]={'idx':[], 'xy':[]} 
            region_idx = int(re.split('(\d+)',i)[1])
            regions_dict[prop_dict[i]]['idx'].append(region_idx)
            region_xy = (int((region_idx-rem(region_idx,width))/(height*width)),rem(region_idx,width))
            regions_dict[prop_dict[i]]['xy'].append(region_xy)        
    else:
        pass #TODO
    #TODO
    '''for idx,region in enumerate(regions_dict):
        for xy in regions_dict[region]['xy']:
            #print(xy)
            rec = Rectangle((xy[1], height - 1- xy[0]), 1, 1, color=color[idx], label=region) #TODO  
            ax.add_patch(rec)
            label_list.append(rec)'''

    '''start_p = Rectangle((start[1], h - 1- start[0]), 1, 1, color=color[0], label='start')
    ax.add_patch(start_p)
    goal_p = Rectangle((goal[1], h - 1- goal[0]), 1, 1, color=color[1], label='goal')
    ax.add_patch(goal_p)
    pickup_p = Rectangle((pickup[1], h - 1- pickup[0]), 1, 1, color=color[2], label='pickup')
    ax.add_patch(pickup_p)
    pp = ()
    pp.append(start_p)
    pp.append(goal_p)
    pp.append(pickup_p)'''
    
    start_p = Rectangle((7, height - 1- 0), 1, 1, color=color[0], label='start')
    ax.add_patch(start_p)
    goal_p = Rectangle((1, height - 1), 1, 1, color=color[1], label='goal')
    ax.add_patch(goal_p)
    pickup_p = Rectangle((6, height - 1- 5), 1, 1, color=color[2], label='pickup')
    ax.add_patch(pickup_p)
    pp = []
    pp.append(start_p)
    pp.append(goal_p)
    pp.append(pickup_p)

    # plot reward regions
    plt.text(7.3, 1.2, '+5',fontsize=15)
# IROS
    '''D1 = Rectangle((1,0), 1, 1, color=color[0], label='D1')
    ax.add_patch(D1)
    D2 = Rectangle((0,4), 1, 1, color=color[1], label='D2')
    ax.add_patch(D2)
    P = Rectangle((6,2), 1, 1, color=color[2], label='P')
    ax.add_patch(P)
    Base = Rectangle((7,7), 1, 1, color=color[3], label='Base')
    ax.add_patch(Base)
    r1 = Rectangle((0,5), 3, 3, color=color[4], label='low reward')
    ax.add_patch(r1)
    r2 = Rectangle((2,0), 2, 1, color=color[4], label='low reward')
    ax.add_patch(r2)
    r3 = Rectangle((1,6), 1, 1, color=color[5], label='high reward')
    ax.add_patch(r3)
    plt.text(1+0.25, h-0.7-7, 'D1', fontsize=15)
    plt.text(0+0.25, h-0.7-3, 'D2', fontsize=15)
    plt.text(7.1, h-0.7, 'Base', fontsize=15)
    plt.text(6+0.4, h-0.7-5, 'P', fontsize=15)
    pp.append(D1)
    pp.append(D2)
    pp.append(P)
    pp.append(Base)
    pp.append(r1)
    pp.append(r3)
    line = Line2D([0], [0], color=color[6], ls='-',lw=2, label='trajectory')
    pp.append(line)
    line = Line2D([0], [0], color='w', marker='o',mfc='k',ms=12,lw=2, label='start')
    pp.append(line)
    line = Line2D([0], [0], color='w', marker='s',mfc='k',ms=12,lw=2, label='end')
    pp.append(line)'''

    '''D1 = Rectangle((2,0), 1, 1, color=color[0], label='D1')
    ax.add_patch(D1)
    D2 = Rectangle((3,7), 1, 1, color=color[1], label='D2')
    ax.add_patch(D2)
    P = Rectangle((0,4), 1, 1, color=color[2], label='P')
    ax.add_patch(P)
    Base = Rectangle((7,7), 1, 1, color=color[3], label='Base')
    ax.add_patch(Base)
    r1 = Rectangle((0,5), 3, 3, color=color[4], label='low reward')
    ax.add_patch(r1)
    r2 = Rectangle((6,0), 2, 1, color=color[4], label='low reward')
    ax.add_patch(r2)
    r3 = Rectangle((7,0), 1, 1, color=color[5], label='high reward')
    ax.add_patch(r3)
    plt.text(2+0.25, h-0.7-7, 'D1', fontsize=15)
    plt.text(3+0.25, h-0.7, 'D2', fontsize=15)
    plt.text(7.1, h-0.7, 'Base', fontsize=15)
    plt.text(0+0.25, h-0.7-3, 'P', fontsize=15)
    pp.append(D1)
    pp.append(D2)
    pp.append(P)
    pp.append(Base)
    pp.append(r1)
    pp.append(r3)
    line = Line2D([0], [0], color=color[6], ls='-',lw=2, label='trajectory')
    pp.append(line)
    line = Line2D([0], [0], color='w', marker='o',mfc='k',ms=12,lw=2, label='start')
    pp.append(line)
    line = Line2D([0], [0], color='w', marker='s',mfc='k',ms=12,lw=2, label='end')
    pp.append(line)'''

    start_marker = Line2D([0], [0], color='w', marker='o',mfc='k',ms=12,lw=2, label='start')
    label_list.append(start_marker)
    end_marker = Line2D([0], [0], color='w', marker='s',mfc='k',ms=12,lw=2, label='end')
    label_list.append(end_marker)

# Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, 30, 1)

    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)

    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        labelbottom=True)

# And a corresponding grid

    ax.grid(color='k',which='both',linewidth=0.5)
    ax.set_xlim([0,width])
    ax.set_ylim([0,height])
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    lgnd=fig.legend(handles=pp,loc='upper right',ncol=1, frameon=False, markerscale=1.05, fontsize=20)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig('/home/xslin/Documents/xslin/research/RL/ConstrainedRL/plot/sim.png', dpi=1000)

    def return_xy(traj_list, scale = 1):
        if traj_list[0] == traj_list[-1]:
            traj_list = traj_list[:-1]
        row_idx = [int((i-i%width)/width) for i in traj_list]
        col_idx = [i%width for i in traj_list]
        x = [i + 0.5 for i in col_idx]
        y = [height - 0.5 - i for i in row_idx]

        dx = scale*(np.array(x[1:]+x[:1])-np.array(x))
        dy = scale*(np.array(y[1:]+y[:1])-np.array(y))

        new_x = np.array(x)+dx
        new_y = np.array(y)+dy

        return x,y,new_x,new_y

    traj_list = []
    if plot_traj == True:
        with open(proj_dir+'/trajectory_log_0.txt', "r") as filestream:
            for f_line in filestream:
                currentline = f_line.split(",")
                traj_list.append([int(i) for i in currentline[:-1]])


        x,y,new_x,new_y = return_xy(traj_list[0])
        global line
        global line_2
        global line_3
        line = plt.plot(x,y,color="blue",label='trajectory')
        line_2 = plt.plot(x[0],y[0],color=color[6],marker='o',mfc='k',ms=10)
        line_3 = plt.plot(x[-1],y[-1],color=color[6],marker='^',mfc='k',ms=10)
        plt.title("Iteration {}".format(1))
        time.sleep(0.5)
        '''arrow = FancyArrowPatch((0,1),(0,1),)
        ar = ax.add_patch(arrow)
        for a,b,c,d in zip(x,y,new_x,new_y):
        #line = a.arrow(a,b,c,d,head_width=0.2,head_length=0.5)
        arrow = FancyArrowPatch((a,b),(c,d))
        ar = ax.add_patch(arrow)'''

        pause = False
        def onClick(event):
            global pause
            pause = not pause
 
        def animate(i):
            if not pause:
                global line
                global line_2
                global line_3
                l = line.pop(0)
                l_2 = line_2.pop(0)
                l_3 = line_3.pop(0)
                l.remove()
                l_2.remove()
                l_3.remove()
                x,y,new_x,new_y = return_xy(traj_list[i])
                line = plt.plot(x,y,color=color[6],label='trajectory')                
                line_2 = plt.plot(x[0],y[0],color=color[6],marker='o',mfc='k',ms=12)
                line_3 = plt.plot(x[-1],y[-1],color=color[6],marker='s',mfc='k',ms=12)
                plt.title("Iteration {}".format(i+1))

        fig.canvas.mpl_connect('button_press_event', onClick)
        ani = FuncAnimation(fig, animate, interval = 500, blit = False, frames = test_iters, repeat=False)
        ani.save(proj_dir+'/animation.gif', writer='imagemagick', fps=2)
        print('Image saved')
    plt.show()

if __name__ == '__main__':
    plot_env('')

