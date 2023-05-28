import numpy as np
import gym
import safety_gym
from PIL import Image
import argparse


# To render rgb: export LD_PRELOAD=""
# To render render regular window: export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so


parser = argparse.ArgumentParser()
parser.add_argument('--env', default="Safexp-PointGoal1-TerminalUnsafe-v0")
args = parser.parse_args()

# for i in gym.envs.registry.all():
#   print(i.id)

env = gym.make(args.env)
env.reset()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_limit = env.action_space.high[0]
print(obs_dim,act_dim,act_limit)

env.render()
# background = env.render(mode='rgb_array', camera_id=1, width=2024,height=2024) # env.viewer._read_pixels_as_in_window()
# blended = background.copy()

start_poss = [None]*100 # [(-4,-4,1), (4,-4,1)]
for start_pos in start_poss:
    # env.start_pos=start_pos
    env.reset()
    for i in range(1000):
        env.render()
        # if i%5 == 0:     
        #     image = env.render(mode='rgb_array', camera_id=1, width=2024,height=2024) #env.viewer._read_pixels_as_in_window()
        #     blended[background!=image] = image[background!=image]   

        action = env.action_space.sample()
        s, r, d, info = env.step(action)
        if d:
            print("DONE!", r, info)
            print("r", r, "s ", s, s.shape, "info ", info)
            break

# Image.fromarray(blended).save('test.pdf')