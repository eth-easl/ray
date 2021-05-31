"""Example of using training on CartPole."""

import argparse

import ray
from ray import tune
from ray.rllib.contrib.alpha_zero.models.custom_torch_models import DenseModel
from ray.rllib.contrib.alpha_zero.environments.cartpole import CartPole
from ray.rllib.models.catalog import ModelCatalog


ray.init(address="auto")

ModelCatalog.register_custom_model("dense_model", DenseModel)

tune.run(
    "contrib/AlphaZero",
     stop={"timesteps_total": 4000},
     max_failures=0,
     config={
        "env": CartPole,
        "num_workers": 7,
        "num_cpus_per_worker": 1,
        "num_gpus": 0,
        "rollout_fragment_length": 50,
        "train_batch_size": 500,
        "sgd_minibatch_size": 64,
        "lr": 1e-4,
        "num_sgd_iter": 1,
        "mcts_config": {
           "puct_coefficient": 1.5,
           "num_simulations": 100,
           "temperature": 1.0,
           "dirichlet_epsilon": 0.20,
           "dirichlet_noise": 0.03,
           "argmax_tree_policy": False,
           "add_dirichlet_noise": True,
        },
        "ranked_rewards": {
           "enable": True,
        },
        "model": {
           "custom_model": "dense_model",
        },
    }
)
