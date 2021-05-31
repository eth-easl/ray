import numpy as np
import pandas as pd

import ray
import ray.autoscaler.sdk
import ray.services

from time import sleep, perf_counter
from tqdm import tqdm
import time
import pandas as pd

num_workers=[1,3,6,12,15,18,24]
runs=10
num_objects=25

ray.init(address="auto")

obj_list_1 = []
ref_list_1 = []

obj_list_2 = []
ref_list_2 = []

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

measurements=[]

def benchmark(obj_size):

    @ray.remote(num_cpus=8)
    class worker_1(object):
        def put_obj(self, sz):

            obj_list = []
            ref_list = []

            for i in range(num_objects):
                a = np.ones(obj_size, dtype=np.uint8)
                #a = 'a' * sz
                obj_list.append(a)

            for i in range(num_objects):
                ref = ray.put(obj_list[i])
                ref_list.append(ref)

            self.obj_list = obj_size
            self.ref_list = ref_list

            #print('done!', self.ref_list)
            return self.ref_list


    @ray.remote
    def get_arr():
        duration=0

        #print(ref_list_1, ref_list_2)
        res = []

        start = time.time()
        for i in range(num_objects):
            #start = time.time()
            #ar = ray.get(ref_list_1[i])
            #end = time.time()
            #res.append((end-start)*1000)
            #if (i==0):
            #    print(ar.size)
        end = time.time()
        duration += end-start

        # start = time.time()
        # for i in range(num_objects):
        #     #start = time.time()
        #     ar = ray.get(ref_list_2[i])
        #     #end = time.time()
        #     #res.append((end-start)*1000)
        #     if (i==0):
        #         print(ar.size)
        # end = time.time()
        # duration += end-start

        sleep(10)
        #return res

        return (duration*1000/(num_objects))


    # Create a remote function with the given resource label attached
    local_get_func = get_arr.options(resources={worker_node_id_1: 0.01})

    w1 = worker_1.remote()
    #w2 = worker_1.remote()

    #local_put_func_1 = put_obj_1.options(resources={worker_node_id_1: 0.01})

    ref = [w.put_obj.remote(obj_size) for w in [w1]]
    [ref_list_1] = ray.get(ref)

    #print(ref_list_1, ref_list_2)
    print('Start benchmarks!')

    sleep(10)

    obj_ref = [local_get_func.remote() for i in range(50)]
    latency = ray.get(obj_ref)
    latency.sort()
    print("obj size: ", obj_size, ", latency(ms): ", latency)
    latency_ar = np.asarray(latency)
    print("avg: ", np.median(latency_ar))
    #np.savetxt("/home/ubuntu/object_store_micro/results/read_remote_latency.csv", latency_ar, delimiter=",")

    #return [obj_size, latency]

# warm-up
#benchmark(10**3)

obj_sizes=[10**3, 10**4]
for o in obj_sizes:
    benchmark(o)
    #measurements.append(benchmark(o))
    sleep(15)

# df = pd.DataFrame(np.asarray(measurements), columns=['obj_size', 'time'])
# print(df)
# df.to_csv('/home/ubuntu/object_store_micro/results/read_objects_remote_np.csv')

