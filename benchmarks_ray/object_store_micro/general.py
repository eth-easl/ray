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
'''
np_list = [np.random.randn(1000000) for i in range(100)]
np_dict = {'weight-' + str(i): np.random.randn(1000000) for i in range(100)},
str_list = [str(i) for i in range(2000000)]
'''

#s = np.ones(2**20, dtype=np.uint8)
s = 'a' * 100*2**20
#feature_list = [str(i) for i in range(2**5)]
#d = pd.DataFrame(0, index=np.arange(2**25), columns=feature_list, dtype=np.uint8)

#s = 'a' * 2**30

#sleep(10)

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

#worker_node_id_1 = nodes[0]
#worker_node_id_2 = nodes[1]

#print(driver_node_id, worker_node_id_1)

@ray.remote(num_cpus=1)
def put_obj():
    start_time = time.time()
    refs=[]

    for i in range(20):
        refs.append(ray.put(s))
        sleep(5)

    end_time = time.time()
    print("Writing took: ",  end_time - start_time, " sec")

#sleep(10)

#print("Read!")

@ray.remote(num_cpus=1)
def get_obj(a):
    print("hello!")
    print(len(a))
    #ray.get(ref)
    #sleep(10)
    t = 'a' * 2**20
    return t

local_put_func = put_obj.options(resources={driver_node_id: 0.01})

ref = local_put_func.remote()
ray.get(ref)

# start_time = time.time()
# obj_ref = [get_obj.remote(s) for i in range(num_workers)]
# results = [ray.get(o) for o in obj_ref]
# #print(results)
# end_time = time.time()
# print("Reading took: ", end_time-start_time)
# sleep(10)

