import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm, rcParams
from matplotlib import rc
import os
import pandas as pd
import seaborn as sns
import argparse

#####################################################################################

parser = argparse.ArgumentParser()
parser.add_argument(
    '--path',
    default='./images',
    help="path"
)
args = parser.parse_args()
#####################################################################################

def plotdata(data, name, xaxis, xlabel, ylabel):
    s = 20
    rc_ = {'figure.figsize':(10,6),'axes.labelsize': 30, 'xtick.labelsize': s, 
           'ytick.labelsize': s, 'legend.fontsize': 20}
    sns.set(rc=rc_, style="darkgrid")
    # rc('text', usetex=True)
    
    fig, ax = plt.subplots()
    
    lw = 2.0
    for (mean, std, label) in data:
        ax.plot(xaxis, mean,  label=label, lw = lw)
        ax.fill_between(xaxis, mean - std, mean + std, alpha=0.4)
    if name=="exp_1":
        plt.axvline(x=-3.4, label="MinMax", linestyle="--") # For exp 1
    
    ax.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
    #ax.ticklabel_format(axis='y',style='scientific', useOffset=True)
    fig.tight_layout()
    fig.savefig("{}/{}.pdf".format(args.path,name), bbox_inches='tight')
    plt.show()

def process_data(alldata, smooth=0):
    pdata = []
    s = 0.5
    o = alldata[0][0].shape[0]
    for (data, label) in alldata:
        # for i in range(o):
        #     data[i] = np.convolve(data[i], np.ones(m)/m, mode='same')

        mean = data.mean(axis=0)
        std = data.std(axis=0)*s

        if smooth:
            mean = np.convolve(mean, np.ones(smooth)/smooth, mode='valid')
            std = np.convolve(std, np.ones(smooth)/smooth, mode='valid')

        pdata.append([mean, std, label])

    return pdata

#####################################################################################

if __name__ == '__main__': 
    name = "exp_1" # E.g "exp_2_returns", "exp_1"
    runs = 4
    
    if name=="exp_1":
        yaxis = []
        plots = ["policy_length", "failure_rate", "timesteps_convergence"]
        for plot in plots:
            data = []
            for run in range(runs): 
                data_path = "data/exp_1.run_{}.npy".format(run)
                if os.path.exists(data_path):
                    log = np.load(data_path, allow_pickle=True).tolist()
                    d = []
                    for l in log:
                        d.append(l[plot])
                    data.append(np.array(d))
                else:
                    print("Failed loading:", data_path)
                    break
            data = np.array(data)
            if plot == "policy_length":
                plot = "Converged timesteps"
            if plot == "failure_rate":
                plot = "Converged failure rate"
            if plot == "cumulative_failures":
                plot = "Cummulative failures"
            if plot == "timesteps_convergence":
                plot = "Total timesteps"
            
            data = (data-data.min())/(data.max()-data.min())
            yaxis.append((data, plot))
            print("LOADED: ", data.shape)
        xaxis = np.linspace(0,-5,41)
        plotdata(process_data(yaxis, smooth=10), name, xaxis, "penalty", "")
    
    if "exp_2" in name:
        yaxis = []
        plots = [0, 1, 2, 3]
        for plot in plots:
            data = []
            for run in range(runs): 
                data_path = "data/{}.run_{}.npy".format(name, run)
                if os.path.exists(data_path):
                    log = np.load(data_path, allow_pickle=True).tolist()
                    data.append(np.array(log[plot]))
                else:
                    print("Failed loading:", data_path)
                    break
            data = np.array(data)
            if plot == 0:
                plot = r"sp=0"
            if plot == 1:
                plot = r"sp=0.25"
            if plot == 2:
                plot = r"sp=0.5"
            if plot == 3:
                plot = r"sp=0.75"
            
            yaxis.append((data, plot))
            print("LOADED: ", data.shape)   
        xaxis = np.arange(data[0][0].shape[0]) 
        plotdata(process_data(yaxis, smooth=200), name, xaxis, "episode", name.split("_")[-1])
