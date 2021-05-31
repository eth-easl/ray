import ray
import gym
from ray import tune
# from ray.rllib.agents import ppo
# from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.logger import pretty_print
import torch

from datetime import datetime as dt

# basic training
ray.init(address='auto')
#ray.init()
#config = ppo.DEFAULT_CONFIG.copy()

# CartPole-v0 config
config={}
config["num_gpus"] = 0
config["num_workers"] = 1
config["num_cpus_per_worker"] = 1
# config["timesteps_per_iteration"] = 20000
# config["placement_strategy"] = "PACK"
# config["gamma"] = 0.99
# config["lr"] = 3e-4
# config["observation_filter"] = "MeanStdFilter"
# config["num_sgd_iter"] = 6
# config["vf_loss_coeff"] = 0.01
config["framework"] = torch
#config["_use_trajectory_view_api"] = False
# config["model"]["fcnet_hiddens"] = [32]
# config["model"]["fcnet_activation"] = "linear"
# config["model"]["vf_share_layers"] = True
# config["eager"] = False

# ppo_agent = PPOTrainer(config=config, env="CartPole-v0")

# # CHECKPOINT_ROOT = "tmp/es/env"

# N_ITER = 10

# for n in range(N_ITER):
#   result = ppo_agent.train()
#   print(pretty_print(result))

#   if (n%5 == 0):
#     checkpoint = ppo_agent.save()
#     print("checkpoint saved at", checkpoint)


# tune hyperparameters (ray.tune)

#config["env"] = "CartPole-v0"
start = dt.now()
tune.run(
    "DDPPO",
    stop={"timesteps_total": 100000},
    config=config
)
end = dt.now()
print("Elapsed Time: {}".format((end-start).total_seconds()))

# compute actions on a trained agent

# # instantiate env class
# env = gym.make('CartPole-v0')
# # run until episode ends
# episode_reward = 0
# done = False
# obs = env.reset()
# while not done:
#     action = es_agent.compute_action(obs)
#     obs, reward, done, info = env.step(action)
#     episode_reward += reward

# print(episode_reward)