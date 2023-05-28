import numpy as np
import hashlib
from collections import defaultdict
import itertools


def V_equal(V1,V2,epsilon=1e-1):  
    diffs = 0  
    for state in V1:
        diff = abs(V1[state]-V2[state])
        diffs += diff
        if diff>epsilon:
            return False
    return True

def to_hash(x, b=False):
    # hash_x = x['mission'] + ' | '+ hashlib.md5(x['image'].tostring()).hexdigest()
    hash_x = hashlib.md5(x.tostring()).hexdigest() if b else x
    return hash_x

def epsilon_greedy_policy_improvement(env, Q, epsilon = 1):
    """
    Implements policy improvement by acting epsilon-greedily on Q

    Arguments:
    env -- environment with which agent interacts
    Q -- Action function for current policy

    Returns:
    policy_improved -- Improved policy
    """

    def policy_improved(state, goal = None, epsilon = epsilon):
        probs = np.ones(env.action_space.n, dtype=float)*(epsilon/env.action_space.n)
        best_action = np.random.choice(np.flatnonzero(Q[state][goal] == Q[state][goal].max())) #np.argmax(Q[state][goal]) #
        probs[best_action] += 1.0 - epsilon
        return probs
    return policy_improved


class MinMaxPenalty():
    """
    Learn the highest reward/penalty that minimises the probability of reaching bad terminal states 
    Usage in training loop: 
        minmaxpenalty = MinMaxPenalty()
        for each step:
            - take an action and get reward and q_value (or just [value] if using policy gradient)
            penalty = minmaxpenalty.update(reward, Q[state])
            if info["unsafe"]:
                reward = penalty
    """
    def __init__(self, rmin=0, rmax=0):
        self.rmin = rmin
        self.rmax = rmax
        self.vmin = self.rmin
        self.vmax = self.rmax
        self.penalty = min([self.rmin, (self.vmin-self.vmax)])
    
    def update(self, reward, Q):
        self.rmin = min([self.rmin, reward])
        self.rmax = max([self.rmax, reward])
        self.vmin = min([self.vmin, self.rmin, min(Q)])
        self.vmax = max([self.vmax, self.rmax, max(Q)])
        self.penalty = min([self.rmin, (self.vmin-self.vmax)])

        return self.penalty


def Safe_Q_learning(env, penalty_type="cons", converge=False, gamma=1, epsilon=1, alpha=1, maxiter=100, mean_episodes=1000, p=True):
    """
    Implements Q_learning

    Arguments:
    env -- environment with which agent interacts
    penalty_type -- ...
    gamma -- discount factor
    alpha -- learning rate
    maxiter -- maximum number of episodes

    Returns:
    Q -- New estimate of Q function
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n)+env.rmax)
    V = Q_V(Q)
    behaviour_policy =  epsilon_greedy_policy_improvement(env, Q, epsilon = epsilon)

    stats = {"R":[], "F":[], "T":[], "P":[]}
    stats["R"].append(0)
    stats["F"].append(0)
    k=0
    t=0    
    penalty = None
    minmaxpenalty = MinMaxPenalty()
    state = env.reset()
    state = to_hash(state)

    stop_cond = lambda k: k < maxiter
    if converge:
        stop_cond = lambda k: True

    while stop_cond(k):
        probs = behaviour_policy(state, epsilon = epsilon)
        action = np.random.choice(np.arange(len(probs)), p=probs)            
        state_, reward, done, info = env.step(action)
        state_ = to_hash(state_)
        
        stats["R"][-1] += reward
        stats["F"][-1] += done and reward<0
        
        if penalty_type=="inf":
            penalty += env.rmin
        if penalty_type=="minmax":
            penalty = minmaxpenalty.update(reward, [Q[state].max()])
        
        if info["unsafe"] and penalty:
            reward = penalty
        
        G = 0 if done else np.max(Q[state_])        
        TD_target = reward + gamma*G
        TD_error = TD_target - Q[state][action]
        Q[state][action] = Q[state][action] + alpha*TD_error

        
        state = state_
        t+=1
        if done:                        
            state = env.reset()
            state = to_hash(state)

            stats["T"].append(t)
            stats["R"].append(0)
            stats["F"].append(0)
            stats["P"].append(penalty)
            t=0
            k+=1
            if p and k%mean_episodes == 0:
                mean_return = np.mean(stats["R"][-mean_episodes-1:-1])
                mean_failure = np.mean(stats["F"][-mean_episodes-1:-1])
                print('Episode: ', k, ' | penalty: ', penalty, ' | Mean return: ', mean_return, ' | Mean failure: ', mean_failure,
                      ' | States: ', len(list(Q.keys())), "| Debug: ", minmaxpenalty.rmin, minmaxpenalty.rmax, minmaxpenalty.vmin, minmaxpenalty.vmax, minmaxpenalty.penalty)
            if converge and k%mean_episodes == 0:
                V_ = Q_V(Q)
                if V_equal(V_, V):
                    break
                else:
                    V = V_
                    
    return Q, stats

#########################################################################################
def Q_P(Q):
    P = defaultdict(lambda: 0)
    for state in Q:
        P[state] = np.argmax(Q[state])
    return P
def Q_V(Q):
    V = defaultdict(lambda: 0)
    for state in Q:
        V[state] = np.max(Q[state]).copy()
    return V
