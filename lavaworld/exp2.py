from GridWorld import *
from library import *
import matplotlib.pyplot as plt
import numpy as np
import sys


gamma = 1
alpha=0.1
epsilon=0.1
maxiter=1000000
goals = [(1,4)]
T_states = [(1,4),(2,4)]
start_position = (3,4)

slip_probs = [0,0.25,0.5,0.75]
data1, data2, data3 = [], [], []
run = sys.argv[1]
for slip_prob in slip_probs:
    print("slip_prob: ", slip_prob, "_______________________________")
    env = GridWorld(goals=goals, T_states=T_states, start_position=None, step_reward=-0.1, wall_reward=-0.1, lava_reward=-0.1, goal_reward=1, slip_prob=slip_prob)
    Q,stats = Safe_Q_learning(env, penalty_type="minmax", epsilon=epsilon, alpha=alpha, maxiter=maxiter, p=True)
    data1.append(stats["R"])
    data2.append(stats["F"])
    data3.append(stats["P"])
np.save("data/exp_2_returns.run_{}".format(run), data1)
np.save("data/exp_2_failures.run_{}".format(run), data2)
np.save("data/exp_2_penalties.run_{}".format(run), data3)
