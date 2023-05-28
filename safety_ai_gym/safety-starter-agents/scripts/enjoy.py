#!/usr/bin/env python

import time
import numpy as np
from safe_rl.utils.load_utils import load_policy
from safe_rl.utils.logx import EpochLogger
import imageio
from PIL import Image
import gym


# save=False --> export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
# save=True --> export LD_PRELOAD=""
# python enjoy.py "../data/data_2023_05_27/unsafe_2_cost25/TRPO_Minmax/good_2023-01-16_11-45-53-trpo_minmax_PointGoal1_s0/" --save blend --seed 42

# Awesome model: python enjoy.py ../data/prelim_safety_results/data/unsafe_1/TRPO_Minmax/2023-05-23_10-32-50-trpo_minmax_PointGoal1_s104  --save gif --episode 4

def run_policy(env, get_action, max_ep_len=None, num_episodes=1, render=True, save=False, seed=None):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :("
    
    saved = 0
    save_success = True
    noseed = seed == None
    seed = seed if not noseed else -1
    while saved<num_episodes: # n < num_episodes:
        seed = seed if not noseed else seed+2
        name = f'../../videos/point_goal1_hard/cpo_s{seed}'
        # env = gym.make("Safexp-PointGoal1-TerminalUnsafe-v0")
        
        env.seed(seed)
        images = []
        logger = EpochLogger()
        camera_id = 1
        if save == "blend":
            env.reset()
            env.render_lidar_markers = False
            background = env.render(mode='rgb_array', camera_id=camera_id, width=2024,height=2024) # env.viewer._read_pixels_as_in_window()
            blended = background.copy()
        env.seed(seed)
            
        o, r, d, ep_ret, ep_cost, ep_goal_met, ep_len, n = env.reset(), 0, False, 0, 0, 0, 0, 0
        
        for step in range(max_ep_len):
            if d:
                break
            if render and not save:
                env.render()
                time.sleep(1e-3)
            if save == "gif":
                images.append(env.render(mode='rgb_array', camera_id=camera_id, width=500,height=500)) # , width=2024,height=2024
            if save == "blend" and ep_len%5==0:
                image = env.render(mode='rgb_array', camera_id=camera_id, width=2024,height=2024) #env.viewer._read_pixels_as_in_window()
                blended[background!=image] = image[background!=image]   
                
            a = get_action(o)
            a = np.clip(a, env.action_space.low, env.action_space.high)
            o, r, d, info = env.step(a)
            ep_ret += r
            ep_cost += info.get('cost', 0)
            ep_goal_met += info['goal_met']
            ep_len += 1

            if d or (ep_len == max_ep_len):
                # logger.store(EpRet=ep_ret, EpCost=ep_cost, EpLen=ep_len)

                if save_success and ep_cost<=0 and ep_goal_met and ep_len <= 1000:
                    print(f"Success, seed {seed}, eplen {ep_len}, goalmet {ep_goal_met}")
                    if ep_len < 100:
                        continue
                    name += "_success"
                    saved += 1
                    if save == "gif":
                        imageio.mimsave(f'{name}.gif', [images[i] for i in range(0,len(images),5)], fps=60)
                    if save == "blend":
                        Image.fromarray(blended).save(f'{name}.pdf')
                    print('Episode %d \t EpRet %.3f \t EpCost %.3f \t EpLen %d'%(seed, ep_ret, ep_cost, ep_len))
                if (not save_success) and (ep_cost>0 or (not ep_goal_met)):
                    print(f"Failure, seed {seed}, eplen {ep_len}, goalmet {ep_goal_met}")
                    name += "_failure"
                    saved += 1
                    if save == "gif":
                        imageio.mimsave(f'{name}.gif', [images[i] for i in range(0,len(images),5)], fps=60)
                    if save == "blend":
                        Image.fromarray(blended).save(f'{name}.pdf')
                    print('Episode %d \t EpRet %.3f \t EpCost %.3f \t EpLen %d'%(seed, ep_ret, ep_cost, ep_len))

        # logger.log_tabular('EpRet', with_min_and_max=True)
        # logger.log_tabular('EpCost', with_min_and_max=True)
        # logger.log_tabular('EpLen', average_only=True)
        # logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=1000)
    parser.add_argument('--episodes', '-n', type=int, default=1)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--save', '-s', type=str, default="")
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    env, get_action, sess = load_policy(args.fpath,
                                        args.itr if args.itr >=0 else 'last',
                                        args.deterministic)
    run_policy(env, get_action, args.len, args.episodes, not(args.norender), args.save, seed=args.seed)
