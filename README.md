# Intro

This repository contains the code to replicate the experiments of the paper "ROSARL: Reward-Only Safe Reinforcement Learning". The paper introduces a new framework for safe RL where the agent learns safe policies solely from scalar rewards using any suitable RL algorithm. This is achieved by replacing the rewards at unsafe terminal states by the minmax penalty, which is the exact highest reward whose optimal policy minimises the probability of reaching unsafe states.

![Trajectories from learned policies of baselines vs ours](algorithms_trajectories.png)


<table style="align-items: center;white-space: nowrap;display: flex;flex-direction: column;">
  <tr>
    <th rowspan="2" scope="rowgroup">TRPO</th>
    <td style="text-align:center"> Success </td>
    <td> <img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/trpo_s1_success.gif"  alt="1" width = auto height = auto ></td>
    <td><img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/trpo_s17_success.gif" alt="2" width = auto height = auto></td>
    <td><img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/trpo_s29_success.gif" alt="2" width = auto height = auto></td>
    <td><img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/trpo_s39_success.gif" alt="2" width = auto height = auto></td>
   </tr> 
   <tr>
    <td style="text-align:center"> Failure </td>
    <td> <img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/trpo_s3_failure.gif"  alt="1" width = auto height = auto ></td>
    <td><img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/trpo_s5_failure.gif" alt="2" width = auto height = auto></td>
    <td><img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/trpo_s7_failure.gif" alt="2" width = auto height = auto></td>
    <td><img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/trpo_s9_failure.gif" alt="2" width = auto height = auto></td>
  </tr>
  <tr>
    <th rowspan="2" scope="rowgroup">TRPO Lagrangian</th>
    <td style="text-align:center"> Success </td>
    <td> <img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/trpo_lagrangian_s1_success.gif"  alt="1" width = auto height = auto ></td>
    <td><img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/trpo_lagrangian_s3_success.gif" alt="2" width = auto height = auto></td>
    <td><img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/trpo_lagrangian_s15_success.gif" alt="2" width = auto height = auto></td>
    <td><img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/trpo_lagrangian_s23_success.gif" alt="2" width = auto height = auto></td>
   </tr> 
   <tr>
    <td style="text-align:center"> Failure </td>
    <td> <img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/trpo_lagrangian_s5_failure.gif"  alt="1" width = auto height = auto ></td>
    <td><img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/trpo_lagrangian_s7_failure.gif" alt="2" width = auto height = auto></td>
    <td><img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/trpo_lagrangian_s9_failure.gif" alt="2" width = auto height = auto></td>
    <td><img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/trpo_lagrangian_s11_failure.gif" alt="2" width = auto height = auto></td>
  </tr>
  <tr>
    <th rowspan="2" scope="rowgroup">CPO</th>
    <td style="text-align:center"> Success </td>
    <td> <img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/cpo_s1_success.gif"  alt="1" width = auto height = auto ></td>
    <td><img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/cpo_s3_success.gif" alt="2" width = auto height = auto></td>
    <td><img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/cpo_s5_success.gif" alt="2" width = auto height = auto></td>
    <td><img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/cpo_s7_success.gif" alt="2" width = auto height = auto></td>
   </tr> 
   <tr>
    <td style="text-align:center"> Failure </td>
    <td> <img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/cpo_s9_failure.gif"  alt="1" width = auto height = auto ></td>
    <td><img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/cpo_s25_failure.gif" alt="2" width = auto height = auto></td>
    <td><img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/cpo_s33_failure.gif" alt="2" width = auto height = auto></td>
    <td><img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/cpo_s35_failure.gif" alt="2" width = auto height = auto></td>
  </tr>
  <tr>
    <th rowspan="2" scope="rowgroup">TRPO Minmax</th>
    <td style="text-align:center"> Success </td>
    <td> <img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/trpo_minmax_s3_success.gif"  alt="1" width = auto height = auto ></td>
    <td><img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/trpo_minmax_s5_success.gif" alt="2" width = auto height = auto></td>
    <td><img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/trpo_minmax_s9_success.gif" alt="2" width = auto height = auto></td>
    <td><img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/trpo_minmax_s13_success.gif" alt="2" width = auto height = auto></td>
   </tr> 
   <tr>
    <td style="text-align:center"> Failure </td>
    <td> <img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/trpo_minmax_s1_failure.gif"  alt="1" width = auto height = auto ></td>
    <td><img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/trpo_minmax_s7_failure.gif" alt="2" width = auto height = auto></td>
    <td><img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/trpo_minmax_s11_failure.gif" alt="2" width = auto height = auto></td>
    <td><img src="safety_ai_gym/videos/point_goal1_terminal_unsafe/trpo_minmax_s25_failure.gif" alt="2" width = auto height = auto></td>
  </tr>
</table>

## Supported RL Algorithms and General Usage

ROSARL is compatible with any RL Algorithm. One can simply [estimate the minmax penalty](learning_minmax_penalty.py) during learning, and replace the environments rewards at unsafe states with it. See [learning_minmax_penalty.py](learning_minmax_penalty.py) for a simple method of estimating the minamx penalty during learning by using the value function being learned by an RL algorithm. 

# Running the Safety Gym experiments

## Installation
These experiments use the [safety gym]([safety_ai_gym/safety-gym](https://github.com/openai/safety-gym)) code (but modified to include environments that terminate when reaching unsafe states), and the [safety starter agents]([safety_ai_gym/safety-starter-agents](https://github.com/openai/safety-starter-agents)) code (but modified to include TRPO-Minmax, which is just TRPO but modified to use the [learned Minmax penalty](learning_minmax_penalty.py)).   

First install [openai mujoco](https://github.com/openai/mujoco-py). Then install the required packages:

```
pip install -r requirements.txt
```

```
cd safety_ai_gym/safety-gym

pip install -e .
```

```
cd safety_ai_gym/safety-starter-agents

pip install -e .
```

## Getting Started

**Example Script:** To run TRPO-Minmax on the `Safexp-PointGoal1-v0` environment from Safety Gym, using neural networks of size (64,64):

```
from safe_rl import trpo_minmax
import gym, safety_gym

trpo_minmax(
	env_fn = lambda : gym.make('Safexp-PointGoal1-TerminalUnsafe-v0'),
	ac_kwargs = dict(hidden_sizes=(64,64))
	)

```


**Reproduce Experiments from Paper:** To reproduce an experiment from the paper, run:

```
cd /path/to/safety-starter-agents/scripts
python experiment.py --algo ALGO --task TASK --robot ROBOT --unsafe_terminate UNSAFE_TERMINATE --seed SEED 
	--exp_name EXP_NAME --cpu CPU
```

where 

* `ALGO` is in `['ppo', 'ppo_lagrangian', 'trpo', 'trpo_lagrangian', 'cpo']`.
* `TASK` is in `['goal1', 'goal2', 'button1', 'button2', 'push1', 'push2']` .
* `ROBOT` is in `['point', 'car', 'doggo']`.
* `UNSAFE_TERMINATE` is in `[0,1,2]`. 0 is the original safety-gym environments that do not terminate when unsafe states are reached. 2 is the modified safety-gym environments that terminate when unsafe states are reached.
* `SEED` is an integer. In the paper experiments, we used seeds of 0, 10, and 20, but results may not reproduce perfectly deterministically across machines.
* `CPU` is an integer for how many CPUs to parallelize across.

`EXP_NAME` is an optional argument for the name of the folder where results will be saved. The save folder will be placed in `/path/to/safety-starter-agents/data`. 


**Plot Results:** Plot results with:

```
cd /path/to/safety-starter-agents/scripts
python plot.py data/path/to/experiment
```

**Watch Trained Policies:** Test policies with:

```
cd /path/to/safety-starter-agents/scripts
python enjoy.py data/path/to/experiment
```

## Cite the Paper

If you use this work, please cite:

```
@article{NangueTasse2023,
    author = {Nangue Tasse, Geraud and Love, Tamlin and Nemecek, Mark and James, Steven and Rosman, Benjamin},
    title = {{ROSARL: Reward-Only Safe Reinforcement Learning}},
    year = {2023}
}
```
