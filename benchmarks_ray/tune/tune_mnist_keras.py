import argparse
from tensorflow.keras.datasets import mnist

from ray.tune.integration.keras import TuneReportCallback
from ray import tune

parser = argparse.ArgumentParser()
parser.add_argument(
    "--smoke-test", action="store_true", help="Finish quickly for testing")
args, _ = parser.parse_known_args()


class TrainMnistClass(tune.Trainable):
    # https://github.com/tensorflow/tensorflow/issues/32159

    def setup(self, config):
        import tensorflow as tf
        self.batch_size = 128
        self.num_classes = 10
        self.epochs = 12

        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(config["hidden"], activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.num_classes, activation="softmax")
        ])

        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.SGD(
                lr=config["lr"], momentum=config["momentum"]),
            metrics=["accuracy"])

    def step(self):

        self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=0,
            # validation_data=(self.x_test, self.y_test),
            # callbacks=[TuneReportCallback({
            #     "mean_accuracy": "accuracy"
            # })]
        )

        d = self.model.evaluate(self.x_test, self.y_test, return_dict=True)
        return {"mean_accuracy": d["accuracy"]}



if __name__ == "__main__":
    import ray
    from ray import tune
    from ray.tune.schedulers import AsyncHyperBandScheduler
    from ray.tune.schedulers import ASHAScheduler

    mnist.load_data()  # we do this on the driver because it's not threadsafe

    #(x_train, y_train), (x_test, y_test) = mnist.load_data()

    ray.init(address="auto")
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", max_t=400, grace_period=20)

    #sched = ASHAScheduler(metric="mean_accuracy", mode="max")

    analysis = tune.run(
        #tune.with_parameters(train_mnist, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test),
        TrainMnistClass,
        name="exp",
        scheduler=sched,
        metric="mean_accuracy",
        mode="max",
        stop={
            "mean_accuracy": 0.99,
            "training_iteration": 5 if args.smoke_test else 300
        },
        num_samples=10,
        resources_per_trial={
            "cpu": 1,
            "gpu": 0
        },
        config={
            "threads": 2,
            "lr": tune.uniform(0.001, 0.1),
            "momentum": tune.uniform(0.1, 0.9),
            "hidden": tune.randint(32, 512),
        })
    print("Best hyperparameters found were: ", analysis.best_config)
