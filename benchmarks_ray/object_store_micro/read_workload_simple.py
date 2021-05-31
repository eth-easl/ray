import numpy as np
import pandas as pd

import ray
import ray.autoscaler.sdk
import ray.services

from time import sleep, perf_counter
from tqdm import tqdm
import time
import pandas as pd

runs=10
run_time = 5

ray.init(address="auto")

obj_list = []
ref_list1 = []
ref_list2 = []
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
worker_node_id_2 = nodes[1]

worker_node_id_1 = 'node:10.138.0.56'
worker_node_id_2 = 'node:10.138.0.57'

print(worker_node_id_1, worker_node_id_2)

# Obtain the local ip address
local_hostname = ray.services.get_node_ip_address()
assert isinstance(local_hostname, str)

 # Ray has a predefined "node id" resource for locality placement
node_id = f"node:{local_hostname}"
print(node_id)
# Check to make sure the node id resource exists
assert node_id in ray.cluster_resources()

measurements=[]


@ray.remote
class worker_1():
    def put_obj(self, sz, num_objects):

        #a = 'a'* sz
        a = np.ones(sz, dtype=np.uint8)
        # d = {}
        # for i in range(300):
        #     k = 'key' + str(i)
        #     d[k] = np.ones(575000, dtype=np.uint8)

        ref_list = []
        #print("hello!!")

        for i in range(num_objects):
            ref = ray.put(a)
            ref_list.append(ref)

        #self.ref_list = ref_list

        #print('done!', self.ref_list)
        return ref_list


def benchmark(obj_size, w):

    #num_objects=int(3*10**9/obj_size)
    #num_objects=min(100, int(10**10/obj_size))
    num_objects=1000

    @ray.remote
    def get_arr():
        #sleep_time = sleep_us/1000000
        op=0
        start = time.time()
        #print("hello!")
        #print(len(ref_list))
        #while (time.time() - start < run_time):
        while (op < num_objects):
            ar = ray.get(ref_list1[op])
            op += 1
            #print("*************************************************")
            #sleep(sleep_time)
            #del ar
        end = time.time()
        #print(end-start)
        return (op, (end - start))

    # Create a remote function with the given resource label attached
    local_worker = worker_1.options(resources={worker_node_id_1: 0.01})
    local_get_func = get_arr.options(resources={worker_node_id_2: 0.01})

    w1 = local_worker.remote()

    ref = [w.put_obj.remote(obj_size, num_objects) for w in [w1]]
    [ref_list1] = ray.get(ref)

    # ref = [w.put_obj.remote(obj_size, num_objects) for w in [w1]]
    # [ref_list2] = ray.get(ref)

    sleep(10)

    # obj = local_get_func.remote()
    # ray.get(obj)
    # sleep(10)

    num_workers = [w]

    for i in num_workers:
        #for s in sleep_us_list:
            print("--------------------- workers: ", w)
            sleep(10)
            obj_ref = [local_get_func.remote() for j in range(i)]

            results = ray.get(obj_ref)

            if (len(results) > 0):

                op_per_worker = [res[0] for res in results]
                time_per_worker = [res[1] for res in results]

                lat_per_worker = [res[1]/res[0] for res in results]
                mean_lat = np.median(np.asarray(lat_per_worker))

                print('mean latency is: ', mean_lat*1000, " ms")

                return mean_lat*1000



# warm-up

obj_sizes=[100]*5
num_workers = [1]
measurements = {}
times=1
for o in obj_sizes:
    measurements[o] = []
    for i  in range(times):
        measurements[o].append(benchmark(o,1))
        sleep(10)
    sleep(30)

for o in obj_sizes:
    print("object size(MB): ", o/10**6, " mean lat: ", np.median(measurements[o]))

#print("mean lat: ", np.median(measurements), "max: ", max(measurements), "min: ", min(measurements))
#df = pd.DataFrame(np.asarray(measurements), columns=['obj_size','Latency(ms)'])
#print(df)
#df.to_csv('/home/ubuntu/object_store_micro/results/read_objects_storage_str.csv')