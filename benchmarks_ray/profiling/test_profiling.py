import numpy as np
import pandas as pd

import ray
import ray.autoscaler.sdk
import ray.services

from time import sleep, perf_counter
from tqdm import tqdm
import time
import pandas as pd
import random


num_workers=1
runs=10
num_objects=10
obj_sizes = np.arange(10**3, 10**4, 1)

ray.init(address="auto")

res = ray.cluster_resources()
res_keys = res.keys()
nodes = []

for e in res_keys:
    if ('node' in e):
        nodes.append(e)

local_hostname = ray.services.get_node_ip_address()
driver_node_id = f"node:{local_hostname}"

# # Check to make sure the node id resource exists
assert driver_node_id in nodes
nodes.remove(driver_node_id)

worker_node_id_1 = nodes[0]
# worker_node_id_2 = nodes[1]

# print(driver_node_id, worker_node_id_1, worker_node_id_2)
@ray.remote(num_cpus=1)
def put_objects():
    print("hello")
    for i in range(num_objects):
        sz = random.choice(obj_sizes)
        ar = np.ones(sz, dtype=np.uint8)
        ray.put(ar)
    sleep(10)


refs = [put_objects.remote() for i in range(num_workers)]
ray.get(refs)


