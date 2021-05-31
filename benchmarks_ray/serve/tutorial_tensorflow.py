# yapf: disable
import ray
# __doc_import_begin__
from ray import serve

import os
import tempfile
import numpy as np
import requests
# __doc_import_end__
# yapf: enable
from time import sleep
import time

# __doc_train_model_begin__
TRAINED_MODEL_PATH = "/home/ubuntu/mnist_model.h5"

@ray.remote(num_cpus=1)
def query_server(num_requests):
    for i in range(num_requests):
        resp = requests.get("http://localhost:8000/mnist", json={"array": np.random.randn(1000 * 1000).tolist()})
        #print(resp.json())
        sleep(0.2)

def train_and_save_model():
    import tensorflow as tf
    # Load mnist dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #x_train, x_test = x_train / 255.0, x_test / 255.0

    print(x_test)

    # Train a simple neural net model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=1)

    model.evaluate(x_test, y_test, verbose=2)
    model.summary()

    # Save the model in h5 format in local file system
    model.save(TRAINED_MODEL_PATH)


if not os.path.exists(TRAINED_MODEL_PATH):
    train_and_save_model()
# __doc_train_model_end__


# __doc_define_servable_begin__
class TFMnistModel:
    def __init__(self, model_path):
        import tensorflow as tf
        self.model_path = model_path
        if not os.path.exists(model_path):
            train_and_save_model()
        # could give the model as an argument
        self.model = tf.keras.models.load_model(model_path)

    async def __call__(self, starlette_request):
        # Step 1: transform HTTP request -> tensorflow input
        # Here we define the request schema to be a json array.
        #print((await starlette_request.json())["array"])
        received_array = np.array((await starlette_request.json())["array"])
        input_array = np.random.randn(28 * 28)
        reshaped_array = input_array.reshape((1, 28, 28))

        # Step 2: tensorflow input -> tensorflow output
        prediction = self.model(reshaped_array)

        # Step 3: tensorflow output -> web output
        return {
            "prediction": prediction.numpy().tolist(),
            "file": self.model_path
        }



ray.init(address="auto")
client = serve.start(http_host="0.0.0.0", )

config = {"num_replicas": 8}
client.create_backend("tf:v1", TFMnistModel, TRAINED_MODEL_PATH, config=config)
client.create_endpoint("tf_classifier", backend="tf:v1", route="/mnist")

num_clients = 10
req_per_client = 1

start = time.time()
refs = [query_server.remote(req_per_client) for i in range(num_clients)]
ray.get(refs)
end = time.time()
duration = end-start

print("Serving took ", duration, " sec.")
print("Total requests: ", num_clients*req_per_client)
print("Throughput: ", (num_clients*req_per_client)/duration, "req/sec")
