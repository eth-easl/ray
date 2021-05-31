from ray.tune.examples.mnist_pytorch import (train, test, get_data_loaders,
                                             ConvNet)
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from torch import optim
import os
import torch

def train_convnet(config, checkpoint_dir=None):
    # Create our data loaders, model, and optmizer.
    step = 0
    train_loader, test_loader = get_data_loaders()
    model = ConvNet()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.get("lr", 0.01),
        momentum=config.get("momentum", 0.9))

    print("Parameters: ", config )

    # If checkpoint_dir is not None, then we are resuming from a checkpoint.
    # Load model state and iteration step from checkpoint.
    if checkpoint_dir:
        print("---------------------------------------------- Loading from checkpoint.")
        path = os.path.join(checkpoint_dir, "checkpoint")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        step = checkpoint["step"]

    while True:
        train(model, optimizer, train_loader)
        acc = test(model, test_loader)
        if step % 5 == 0:
            # Every 5 steps, checkpoint our current state.
            # First get the checkpoint directory from tune.
            with tune.checkpoint_dir(step=step) as checkpoint_dir:
                # Then create a checkpoint file in this directory.
                path = os.path.join(checkpoint_dir, "checkpoint")
                # Save state to checkpoint file.
                # No need to save optimizer for SGD.
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "mean_accuracy": acc
                }, path)
        step += 1
        tune.report(mean_accuracy=acc)

scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=5,
        hyperparam_mutations={
            # distribution for resampling
            "lr": lambda: np.random.uniform(0.0001, 1),
            # allow perturbations within this set of categorical values
            "momentum": [0.8, 0.9, 0.99],
        })


class CustomStopper(tune.Stopper):
        def __init__(self):
            self.should_stop = False

        def __call__(self, trial_id, result):
            max_iter = 5
            if not self.should_stop and result["mean_accuracy"] > 0.96:
                self.should_stop = True
            return self.should_stop or result["training_iteration"] >= max_iter

        def stop_all(self):
            return self.should_stop

stopper = CustomStopper()

ray.init(address="auto")

analysis = tune.run(
        train_convnet,
        name="pbt_test",
        scheduler=scheduler,
        metric="mean_accuracy",
        mode="max",
        verbose=1,
        stop=stopper,
        # export_formats=[ExportFormat.MODEL],
        checkpoint_score_attr="mean_accuracy",
        keep_checkpoints_num=4,
        num_samples=4,
        config={
            "lr": tune.uniform(0.001, 1),
            "momentum": tune.uniform(0.001, 1),
        })

print("Best hyperparameters found were: ", analysis.best_config)
