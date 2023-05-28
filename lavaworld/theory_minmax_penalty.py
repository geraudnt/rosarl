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
rmin, rmax = -1, 0
p = 0.1
D, C = 2, max([(1-p)-p,p-(1-p)]) # 0.72
minmax = np.min([rmin, (rmin-rmax)*(D/C)])
penalty = np.linspace(-5,0,6)
print(penalty)

states = 4
P = np.zeros((penalty.shape[0], states, states, 2)) # p, S, S, A
P[:,3,3,:] = 1.0
P[:,1,1,:] = 1.0
P[:,2,2,:] = np.array([p,p])
P[:,2,3,:] = np.array([1-p,1-p])
P[:,0,2,:] = np.array([1-p,p])
P[:,0,1,:] = np.array([p,1-p])
R = rmin*np.ones((penalty.shape[0], states, states, 2)) # p, S, S, A
R[:,[1,3],:,:] = 0.0
R[:,0,1,0] = penalty
R[:,0,1,1] = penalty
V = np.zeros((penalty.shape[0], states)) # p, S
pi = np.zeros((penalty.shape[0], states)) # p, S

convergence = np.zeros(penalty.shape[0])
step=0
while True:
    step+=1
    V_pre = V.copy()
    for s in range(states):
        Vs = np.array([(np.array([P[:,s,s_,a]*(R[:,s,s_,a] + V[:,s_]) for s_ in range(states)]).sum(axis=0)) for a in range(2)]).max(axis=0)
        V[:,s] = Vs # V[:,s] + 0.001*(Vs-V[:,s])
    for i in range(penalty.shape[0]): 
        if np.abs(V_pre[i]-V[i]).max() < 1e-4 and convergence[i] == 0:
            convergence[i] = step
    # print(np.abs(V_pre-V).max())
    if np.abs(V_pre-V).max() < 1e-4:
        break
for s in range(states):
    pi[:,s] = np.array([(np.array([P[:,s,s_,a]*(R[:,s,s_,a] + V[:,s_]) for s_ in range(states)]).sum(axis=0)) for a in range(2)]).argmax(axis=0)

convergence = (convergence - convergence.min())/(convergence.max() - convergence.min())
success = np.zeros(penalty.shape[0])
for i in range(penalty.shape[0]):
    if pi[i,0] == 0:
        success[i] = (1-p)
    else:
        success[i] = p

#####################################################################################
lw = 5.0
fig, ax = plt.subplots()
ax.plot(penalty, 1-success, label=r'Failure rate', marker="o", ms = 20, lw = lw)
ax.plot(penalty, convergence, label=r'Total timesteps', marker="o", ms = 20, lw = lw)
plt.axvline(x=minmax, color="black", label=r"Minmax", linestyle="--", lw = lw) 

ax.legend()
plt.xlabel("Penalty")
fig.tight_layout()
fig.savefig("{}/{}.pdf".format(args.path,"convergence"), bbox_inches='tight')
plt.show()
#####################################################################################

p_ = np.linspace(0,1,1000)
p = p_**1
# delta_p_s0 = (1-p)*(1-p) - p*(1-p)
# delta_p_s0_ = p*(1-p) - (1-p)*(1-p)
delta_p_s0 = (1-p) - p
delta_p_s0[-1] = 0
delta_p_s0_ = p - (1-p)
delta_p_s0_[-1] = 0
delta_p_s0_c = np.max([delta_p_s0,delta_p_s0_], axis=0)
delta_p_s2 = (1-p) - (1-p)
C = delta_p_s0_c # np.min([delta_p_s0_c,delta_p_s2], axis=0)

#####################################################################################

lw = 10.0
fig, ax = plt.subplots()
# ax.plot(p_, delta_p_s0,  label=r'$s_0$', lw = lw)
# ax.plot(p_, delta_p_s2,  label=r'$s_2$', lw = lw)
ax.plot(p_, delta_p_s0,  label=r'$\Delta P_{s_0}(\pi_1, \pi_2)$', lw = lw)
ax.plot(p_, delta_p_s0_,  label=r'$\Delta P_{s_0}(\pi_2, \pi_1)$', lw = lw)
ax.plot(p_, delta_p_s2,  label=r'$\Delta P_{s_2}(\pi_1, \pi_2)$', lw = lw)
ax.plot(p_, C,  label=r'$C$', color="b", linestyle="dashed", lw = lw)

ax.legend()
plt.xlabel("p")
plt.ylabel(r'$\Delta P_s$')
# ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
#ax.ticklabel_format(axis='y',style='scientific', useOffset=True)
fig.tight_layout()
fig.savefig("{}/{}.pdf".format(args.path,"controllability"), bbox_inches='tight')
plt.show()
#####################################################################################

p_ = p_[:-1]
p = p[:-1]
C = C[:-1]

P = np.zeros((p.shape[0], 4, 4, 2)) # p, S, S, A
P[:,3,3,:] = 1.0
P[:,1,1,:] = 1.0
P[:,2,2,0] = p
P[:,2,2,1] = p
P[:,2,3,0] = 1-p
P[:,2,3,1] = 1-p
P[:,0,2,0] = 1-p
P[:,0,2,1] = p
P[:,0,1,0] = p
P[:,0,1,1] = 1-p
R = np.ones((p.shape[0], 4, 4, 2)) # p, S, A
R[:,[1,3],:,:] = 0.0
V = np.zeros((p.shape[0], 4)) # p, S
while True:
    V_pre = V.copy()
    for s in range(4):
        V[:,s] = np.array([(np.array([P[:,s,s_,a]*(R[:,s,s_,a] + V[:,s_]) for s_ in range(4)]).sum(axis=0)) for a in range(2)]).max(axis=0)
    # print(np.abs(V_pre-V).max())
    if np.abs(V_pre-V).max() < 1e-1:
        break
D = V.max(axis=1)

a = (rmin) + np.zeros(p.shape[0])
b = (rmin-rmax)*(D/C)
penalty = np.min([a, b], axis=0)
b = (rmin-rmax)*D
penalty1 = np.min([a, b], axis=0)
b = (rmin-rmax)*(1/C)
penalty2 = np.min([a, b], axis=0)

#####################################################################################
lw = 5.0
fig, ax = plt.subplots()
ax.plot(p_, penalty1, label=r'$\min \{\bar R_{MIN}, (\bar R_{MIN} - \bar R_{MAX})D\}$', lw = lw)
ax.plot(p_, penalty2, label=r'$\min \{\bar R_{MIN}, (\bar R_{MIN} - \bar R_{MAX})\frac{1}{C}\}$', lw = lw)
ax.plot(p_, penalty, label=r'$\min \{\bar R_{MIN}, (\bar R_{MIN} - \bar R_{MAX})\frac{D}{C}\}$', lw = lw)
plt.axvline(x=0.5, ymax=0.95, color="black", label=r"$C=0$", linestyle="--", lw = lw) 
plt.axvline(x=1.0, ymax=0.95, color="black", linestyle="--", lw = lw) 

ax.legend()
plt.xlabel("p")
plt.ylabel(r'Penalty')
plt.ylim(-50, 0)
fig.tight_layout()
fig.savefig("{}/{}.pdf".format(args.path,"penalty"), bbox_inches='tight')
plt.show()
#####################################################################################

P = np.zeros((p.shape[0], 4, 4, 2)) # p, S, S, A
P[:,3,3,:] = 1.0
P[:,1,1,:] = 1.0
P[:,2,2,0] = p
P[:,2,2,1] = p
P[:,2,3,0] = 1-p
P[:,2,3,1] = 1-p
P[:,0,2,0] = 1-p
P[:,0,2,1] = p
P[:,0,1,0] = p
P[:,0,1,1] = 1-p
R = rmin*np.ones((p.shape[0], 4, 4, 2)) # p, S, A
R[:,[1,3],:,:] = 0.0


R[:,0,1,:] = np.array([penalty,penalty]).T
V = np.zeros((p.shape[0], 4)) # p, S
pi = np.zeros((p.shape[0], 4)) # p, S
R1 = R.copy()
R1[:,0,1,:] = np.array([penalty1,penalty1]).T
V1 = np.zeros((p.shape[0], 4)) # p, S
pi1 = np.zeros((p.shape[0], 4)) # p, S
R2 = R.copy()
R2[:,0,1,:] = np.array([penalty2,penalty2]).T
V2 = np.zeros((p.shape[0], 4)) # p, S
pi2 = np.zeros((p.shape[0], 4)) # p, S
while True:
    V_pre = V.copy()
    V_pre1 = V1.copy()
    V_pre2 = V2.copy()
    for s in range(4):
        V[:,s] = np.array([(np.array([P[:,s,s_,a]*(R[:,s,s_,a] + V[:,s_]) for s_ in range(4)]).sum(axis=0)) for a in range(2)]).max(axis=0)
        V1[:,s] = np.array([(np.array([P[:,s,s_,a]*(R1[:,s,s_,a] + V1[:,s_]) for s_ in range(4)]).sum(axis=0)) for a in range(2)]).max(axis=0)
        V2[:,s] = np.array([(np.array([P[:,s,s_,a]*(R2[:,s,s_,a] + V2[:,s_]) for s_ in range(4)]).sum(axis=0)) for a in range(2)]).max(axis=0)
    # print(np.abs(V_pre-V).max(), np.abs(V_pre1-V1).max(), np.abs(V_pre2-V2).max())
    if np.abs(V_pre-V).max() < 1e-1 and np.abs(V_pre1-V1).max() < 1e-1 and np.abs(V_pre2-V2).max() < 1e-1:
        break
for s in range(4):
    pi[:,s] = np.array([(np.array([P[:,s,s_,a]*(R[:,s,s_,a] + V[:,s_]) for s_ in range(4)]).sum(axis=0)) for a in range(2)]).argmax(axis=0)
    pi1[:,s] = np.array([(np.array([P[:,s,s_,a]*(R1[:,s,s_,a] + V1[:,s_]) for s_ in range(4)]).sum(axis=0)) for a in range(2)]).argmax(axis=0)
    pi2[:,s] = np.array([(np.array([P[:,s,s_,a]*(R2[:,s,s_,a] + V2[:,s_]) for s_ in range(4)]).sum(axis=0)) for a in range(2)]).argmax(axis=0)

success = np.zeros(p.shape[0])
failure = np.zeros(p.shape[0])
success1 = np.zeros(p.shape[0])
failure1 = np.zeros(p.shape[0])
success2 = np.zeros(p.shape[0])
failure2 = np.zeros(p.shape[0])
for i in range(p.shape[0]):
    if pi[i,0] == 0:
        success[i] = (1-p[i])#*(1-p[i])
        failure[i] = p[i]
    else:
        success[i] = p[i]#*(1-p[i])
        failure[i] = 1-p[i]
    if pi1[i,0] == 0:
        success1[i] = (1-p[i])#*(1-p[i])
        failure1[i] = p[i]
    else:
        success1[i] = p[i]#*(1-p[i])
        failure1[i] = 1-p[i]
    if pi2[i,0] == 0:
        success2[i] = (1-p[i])#*(1-p[i])
        failure2[i] = p[i]
    else:
        success2[i] = p[i]#*(1-p[i])
        failure2[i] = 1-p[i]

#####################################################################################
lw = 5.0
fig = plt.figure()
gs = fig.add_gridspec(3, hspace=0.1)
ax = gs.subplots(sharex=True, sharey=True)
ax[0].plot(p_, success1, label=r'Success $(D)$', lw = lw)
ax[0].plot(p_, failure1, label=r'Failure $(D)$', linestyle="--", lw = lw)
ax[1].plot(p_, success2, label=r'Success  $(\frac{1}{C})$', lw = lw)
ax[1].plot(p_, failure2, label=r'Failure $(\frac{1}{C})$', linestyle="--", lw = lw)
ax[2].plot(p_, success, label=r'Success $(\frac{D}{C})$', lw = lw)
ax[2].plot(p_, failure, label=r'Failure $(\frac{D}{C})$', linestyle="--", lw = lw)
for a in ax:
    a.label_outer()
    a.legend()

plt.xlabel("p")
fig.tight_layout()
fig.savefig("{}/{}.pdf".format(args.path,"success_failure_rates"), bbox_inches='tight')
plt.show()