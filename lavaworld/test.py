from GridWorld import *
from library import *
import matplotlib.pyplot as plt


gamma = 1
maxiter=1000000
alpha=0.1 # 0.25
epsilon=0.1

goals = [(1,4)]
T_states = [(1,4),(2,4)]
start_position = (3,4)
env = GridWorld(goals=goals, T_states=T_states, start_position=None, step_reward=-0.1, wall_reward=-0.1, lava_reward=-5, goal_reward=1, slip_prob=0.25)
fig = env.render()
fig.savefig("images/lavaworld.pdf", bbox_inches='tight')

# Q,stats1 = Safe_Q_learning(env, penalty_type="", epsilon=epsilon, alpha=alpha, converge=1, maxiter=maxiter, p=True)
# # env.render(Q=Q)
# # plt.show()
# env.render( P=Q_P(Q), V = Q_V(Q))
# plt.show()