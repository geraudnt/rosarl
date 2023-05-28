from safe_rl import ppo_lagrangian, ppo_minmax
import gym, safety_gym
from safe_rl.utils.run_utils import setup_logger_kwargs


num_steps = 1e7
steps_per_epoch = 30000
epochs = int(num_steps / steps_per_epoch)

exp_name = "ppo_minmax_point_goal1"
logger_kwargs = setup_logger_kwargs(exp_name, 0)

ppo_minmax(
	env_fn = lambda : gym.make('Safexp-PointGoal1-v0'),
	ac_kwargs = dict(hidden_sizes=(64,64)),
    logger_kwargs=logger_kwargs
	)
