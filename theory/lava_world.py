from turtle import color
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm, rcParams
from matplotlib import rc
import os
import pandas as pd
import seaborn as sns
import argparse


s = 20
# 'figure.figsize':(12,8)
# 'figure.figsize':(10,6)
rc_ = {'figure.figsize':(12,8),'axes.labelsize': 30, 'xtick.labelsize': s, 
        'ytick.labelsize': s, 'legend.fontsize': 20}
sns.set(rc=rc_, style="darkgrid")
# rc('text', usetex=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--path',
    default='./images',
    help="path"
)
args = parser.parse_args()

# Vary p [0 1], Vary n [0 2] in p^n, Vary rmin [-5 0) 

# #####################################################################################
rmin, rmax = -0.1, 1
p = 0.25
D, C = 2, p
minmax = np.min([rmin, (rmin-rmax)*(D/C)])
penalty = np.linspace(-5,0,6)
print("penalty:", penalty)

states = 11
P = np.zeros((penalty.shape[0], states, states, 2)) # p, S, S, A
P[:,8,8,:] = 1.0
P[:,9,9,:] = 1.0
P[:,0,0,0] = 1-p
P[:,0,1,0] = p/2
P[:,0,3,0] = p/2
P[:,0,0,1] = p/2
P[:,0,1,1] = p/2
P[:,0,3,1] = 1-p
P[:,0,0,2] = p/2
P[:,0,1,2] = p/2
P[:,0,3,2] = 1-p
R = np.ones((penalty.shape[0], states, states, 2)) # p, S, S, A
R[:,[1,3],:,:] = 0.0
V = np.zeros((penalty.shape[0], states)) # p, S
while True:
    V_pre = V.copy()
    for s in range(4):
        V[:,s] = np.array([(np.array([P[:,s,s_,a]*(R[:,s,s_,a] + V[:,s_]) for s_ in range(states)]).sum(axis=0)) for a in range(2)]).max(axis=0)
    # print(np.abs(V_pre-V).max())
    if np.abs(V_pre-V).max() <= 0:
        break
D = V.max(axis=1)
print("D",D)