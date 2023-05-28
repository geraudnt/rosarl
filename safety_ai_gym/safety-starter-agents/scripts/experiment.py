#!/usr/bin/env python
import gym 
import safety_gym
import safe_rl
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork


def main(robot, task, unsafe_terminate, algo, seed, exp_name, cpu):

    # Verify experiment
    unsafe_terminate_list = [0,1,2]
    robot_list = ['point', 'car', 'doggo']
    task_list = ['goal1', 'goal2', 'button1', 'button2', 'push1', 'push2']
    algo_list = ['ppo', 'ppo_minmax', 'ppo_lagrangian', 'trpo', 'trpo_minmax', 'trpo_lagrangian', 'cpo']

    algo = algo.lower()
    task = task.capitalize()
    robot = robot.capitalize()
    assert algo in algo_list, "Invalid algo"
    assert task.lower() in task_list, "Invalid task"
    assert robot.lower() in robot_list, "Invalid robot"
    assert unsafe_terminate in unsafe_terminate_list, "Invalid unsafe_terminate"

    # Hyperparameters
    exp_name = algo + '_' + robot + task
    if robot=='Doggo':
        num_steps = 1e8
        steps_per_epoch = 60000
    else:
        num_steps = 1e7
        steps_per_epoch = 30000
    epochs = int(num_steps / steps_per_epoch)
    save_freq = 50
    target_kl = 0.01
    cost_lim = 25
    if unsafe_terminate==2:
        cost_lim = 0
        

    # Fork for parallelizing
    mpi_fork(cpu)

    # # Prepare Logger
    # exp_name = exp_name or (algo + '_' + robot.lower() + task.lower())
    # logger_kwargs = setup_logger_kwargs(exp_name, seed)

    # Algo and Env
    algo = eval('safe_rl.'+algo)
    env_name = 'Safexp-'+robot+task+'-v0'
    if unsafe_terminate == 1:
        env_name = 'Safexp-'+robot+task+'-FlagUnsafe-v0'
    if unsafe_terminate == 2:
        env_name = 'Safexp-'+robot+task+'-TerminalUnsafe-v0'
    env = gym.make(env_name)
    logger_kwargs = setup_logger_kwargs(env_name, seed)

    algo(env_fn=lambda: env,
         ac_kwargs=dict(
             hidden_sizes=(256, 256),
            ),
         epochs=epochs,
         steps_per_epoch=steps_per_epoch,
         save_freq=save_freq,
         target_kl=target_kl,
         cost_lim=cost_lim,
         seed=seed,
         logger_kwargs=logger_kwargs
         )



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='Point')
    parser.add_argument('--task', type=str, default='Goal1')
    parser.add_argument('--algo', type=str, default='ppo')
    parser.add_argument('--unsafe_terminate', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--cpu', type=int, default=1)
    args = parser.parse_args()
    exp_name = args.exp_name if not(args.exp_name=='') else None
    main(args.robot, args.task, args.unsafe_terminate, args.algo, args.seed, exp_name, args.cpu)