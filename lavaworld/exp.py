from GridWorld import *
from library import *
import matplotlib.pyplot as plt
import numpy as np


gamma = 1
alpha=0.1
epsilon=0.1
goals = [(1,10)]
T_states = [(1,10),(3,10),(4,10),(5,10)]
start_position = (7,10)

plots = ["policy_length", "cumulative_failures", "failure_rate", "timesteps_convergence"]
penalties = [-0.1,-1,-10,-100]
data = []
run = 0
for penalty in penalties:
    env = GridWorld(goals=goals, T_states=T_states, start_position=None, step_reward=-0.1, wall_reward=-0.1, lava_reward=penalty, goal_reward=1, slip_prob=0.25)
    Q,stats = Safe_Q_learning(env, cost_type="", epsilon=epsilon, alpha=alpha, converge=1, p=False)
    data.append[stats]
np.save("data/exp1_{}".format(run))