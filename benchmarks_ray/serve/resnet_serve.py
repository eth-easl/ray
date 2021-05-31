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

import PIL
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications import resnet50
import ray.services


# __doc_train_model_begin__
TRAINED_MODEL_PATH = "/home/ubuntu/mnist_model.h5"

@ray.remote(num_cpus=1)
def query_server(num_requests):
    for i in range(num_requests):
        if (i%50 == 0):
            print(i)
        resp = requests.get("http://localhost:8000/resnet", json={"array": np.random.randn(224*224*3).tolist()})
        #print(resp.json())
        #sleep(20)


# __doc_define_servable_begin__
class ResNet50Model:
    def __init__(self, model_path):
        resnet_model = resnet50.ResNet50(weights = 'imagenet')
        self.model = resnet_model
        print("hello!")

    async def __call__(self, starlette_request):
        # Step 1: transform HTTP request -> tensorflow input
        # Here we define the request schema to be a json array.
        #print((await starlette_request.json())["array"])
        received_array = np.array((await starlette_request.json())["array"])
        #received_array = np.random.randn(224 * 224 * 3)
        reshaped_array = received_array.reshape((1,224,224,3))

        # Step 2: tensorflow input -> tensorflow output
        prediction = self.model.predict(reshaped_array)
        #print(prediction.size)

        # Step 3: tensorflow output -> web output
        return {
            "prediction": prediction.tolist(),
        }



ray.init(address="auto")

local_hostname = ray.services.get_node_ip_address()
driver_node_id = f"node:{local_hostname}"

res = ray.cluster_resources()
res_keys = res.keys()
nodes = []

for e in res_keys:
    if ('node' in e):
        nodes.append(e)

assert driver_node_id in nodes
nodes.remove(driver_node_id)
worker_node_id_1 = nodes[0]
worker_node_id_2 = nodes[1]

client = serve.start(http_host="0.0.0.0", )

config = {"num_replicas": 1}
client.create_backend("tf:v1", ResNet50Model, TRAINED_MODEL_PATH, config=config)
client.create_endpoint("tf_classifier", backend="tf:v1", route="/resnet")

num_clients = 2
req_per_client = 500

query_server_local = query_server.options(resources={driver_node_id: 0.01})

start = time.time()
refs = [query_server_local.remote(req_per_client) for i in range(num_clients)]
ray.get(refs)
end = time.time()
duration = end-start

print("Serving took ", duration, " sec.")
print("Total requests: ", num_clients*req_per_client)
print("Throughput: ", (num_clients*req_per_client)/duration, "req/sec")
