# Purpose
```
TODO: This section is taken from the previous readme. Update for current work.
```
We consider a delivery drone that is supposed to achieve pick-up and delivery tasks that arrive stochastically during a mission. Since a delivery drone is often equipped with a camera, it can also gather useful information by monitoring the environment during the pick-up and delivery task. Motivated by the multi-use of drones, we address a persistent monitoring problem where a drone’s high-level decision making is modeled as a Markov decision process (MDP) with unknown transition probabilities. The reward function is designed based on the valuable information over the environment, and the pick-up and delivery tasks are defined by bounded temporal logic specifications. We use a reinforcement learning (RL) algorithm that maximizes the expected sum of rewards while various dynamically arriving temporal logic specifications are satisfied with a desired probability in every episode during learning.

# Setup
The code in this repository is written and tested using Python 2.7 and Ubuntu 20.04. See below sections for Ubuntu and Mac OSX setup instructions.

## Ubuntu
If you do not have a Python 2.7 installation you can (hopefully) still get it from your package manager. On Ubuntu 20.04:
```
sudo apt install python2.7
```
Now you will need to install pip for python 2.7. On Ubuntu 20.04, this is no longer available from the package manager. You can instead install it directly from PyPA using the following:
```
wget https://bootstrap.pypa.io/pip/2.7/get-pip.py
python2 get-pip.py
```
Next you have to install the dependencies. This can be done using the included `requirements.txt`.
```
python2 -m pip install -r requirements.txt
```

## Max OSX
The working document for Mac setup instructions:
https://docs.google.com/document/d/1Liqc0fYq_BstlCLVsLB2QN50RR047NFMK-fj47NKnas/edit

# Usage
Running `main.py` in the `src` directory will use Q-learning to find the optimal policy for a simple pick-up and delivery problem with static rewards. The code will print out some time performance information and, at the end, the results of testing the found optimal policy. Running this code will also make some files in the `output` directory. You can look at `mdp_trajectory_log.txt` with an ANSI escape code compatible text viewer (I use `less -R`) for some colors that represent what is happening during learning. You can find at the top of `main.py` what each color represents. 

Next you can run `plot_route.py` which will show a nice animation of a route generated during policy test.

You can edit the problem and learning parameters in `config/default.yaml`. The animation is not connected to this config file yet, so you will have to edit that file if you want to animate a different problem.

# PyTWTL
PyTWTL is the included program used to generate a DFA from a TWTL specification. You can learn more about PyTWTL and its license in `readme_pytwtl.md`.