import ray
import gym
from ray import tune
from ray.rllib.agents import es
from ray.rllib.agents.es import ESTrainer
from ray.tune.logger import pretty_print

from datetime import datetime as dt

# basic training
ray.init(address='auto')
config = es.DEFAULT_CONFIG.copy()

start = dt.now()

workers=[4, 8, 16, 24, 32]

# CartPole-v0 config
# config["num_gpus"] = 0
# config["num_workers"] = 32
# config["episodes_per_batch"] = 50
# config["num_cpus_per_worker"] = 1
# config["noise_size"] = 25000000

# es_agent = ESTrainer(config=config, env="CartPole-v0")

# # CHECKPOINT_ROOT = "tmp/es/env"

# N_ITER = 100

# for n in range(N_ITER):
#   result = es_agent.train()
#   print(pretty_print(result))

#   if (n%10 == 0):
#     checkpoint = es_agent.save()
#     print("checkpoint saved at", checkpoint)

# end = dt.now()
# print("Elapsed Time: {}".format((end-start).total_seconds()))


# tune hyperparameters (ray.tune)
# here it will run 1 sample (no hyperparameters to choose - default), and it will stop at 'timesteps_total'

# TODO:
# 1)  try: PPO, DD-PPO
# 2) try large bum_workers without num_cpus_per_worker

start = dt.now()
tune.run(
    "ES",
    queue_trials=True,
    stop={"timesteps_total": 5000000},
    config={
        "env": "CartPole-v0",
        "log_level": "WARN",
        "num_gpus": 0,
        "num_workers": 31,
        "episodes_per_batch": 50,
        "noise_size" : 25000000,
        "num_cpus_per_worker" : 1,
        "timesteps_per_iteration": 20000,
        "placement_strategy": "PACK"
    },
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