import numpy as np
import pandas as pd

import ray
import ray.autoscaler.sdk

from time import sleep, perf_counter
from tqdm import tqdm
import time
import ray.services

num_workers=4

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
print(driver_node_id, worker_node_id_1)


@ray.remote
def test():

    s = 'a' * 10*2**20
    objects = 1000

    refs=[]
    start = time.time()
    for i in range(objects):
        refs.append(ray.put(s))
    end = time.time()
    print("Writing took: ", end-start, " sec")


    sleep(10)


    print("Start reading!")
    #obj = []

    start = time.time()
    for i in range(objects):
        obj = ray.get(refs[i])


    end = time.time()
    print("Reading took: ", end-start, " sec")


@ray.remote(num_cpus=1)
def put_obj():
    start_time = time.time()
    refs=[]

    for i in range(20):
        refs.append(ray.put(s))
        sleep(5)

    end_time = time.time()
    print("Writing took: ",  end_time - start_time, " sec")

@ray.remote(num_cpus=1)
def get_obj(a):
    print("hello!")
    print(len(a))
    #ray.get(ref)
    #sleep(10)
    t = 'a' * 2**20
    return t


local_test_func = test.options(resources={worker_node_id_1: 0.01})
ref = local_test_func.remote()
ray.get(ref)
