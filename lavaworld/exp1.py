from GridWorld import *
from library import *
import matplotlib.pyplot as plt
import numpy as np
import sys


gamma = 1
alpha=0.1 # 0.25
epsilon=0.1
goals = [(1,4)]
T_states = [(1,4),(2,4)]
start_position = (3,4)

def evaluateQ(env, Q, gamma=1):
    steps=0
    failures=0   
    for _ in range(100):
        state = env.reset(start_position)
        while True:
            action = Q[state].argmax()
            state, reward, done, _ = env.step(action) 
            steps += 1
            if done:
                failures += reward<0
                break
    return steps/100, failures/100

penalties = np.linspace(0,-5,50) # [-0.1,-0.5,-1,-1.5,-2]#,-2.5,-3,-3.5,-4,-4.5,-5]#,-6,-7,-8,-9,-10]
data = []
run = sys.argv[1]
for penalty in penalties:
    print("Penalty: ", penalty, "_______________________________")
    env = GridWorld(goals=goals, T_states=T_states, start_position=None, step_reward=-0.1, wall_reward=-0.1, lava_reward=penalty, goal_reward=1, slip_prob=0.25)
    Q,stats = Safe_Q_learning(env, penalty_type="", epsilon=epsilon, alpha=alpha, converge=1, p=False)
    policy_length, failure_rate = evaluateQ(env, Q, gamma=1)
    stats = {"policy_length": policy_length, "cumulative_failures": np.sum(stats["F"]), "failure_rate": failure_rate, "timesteps_convergence": np.sum(stats["T"])}
    data.append(stats)
np.save("data/exp_1.run_{}".format(run), data)
