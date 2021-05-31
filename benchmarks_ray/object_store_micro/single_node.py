import numpy as np

import ray
import ray.autoscaler.sdk

from time import sleep, perf_counter
from tqdm import tqdm
import time
import pandas as pd
import ray.services

from ray.internal import free

ARRAY_SIZE = 2*10**8
num=6
num_workers=64
#runs=50
run_time = 1

#a = np.ones(ARRAY_SIZE, dtype=np.uint8)
df= pd.DataFrame()
measurements =[]

@ray.remote(num_cpus=1)
def put_objects():
    #refs=[]
    for i in range(1):
        a = np.ones(10**i, dtype=np.uint8)
        start = time.time()
        for j in range(runs):
            ref = ray.put(a)
        end = time.time()
        duration = (end-start)/runs
        #duration = end-start
        size_gb = 10**i/10**9
        #print(10**i, " bytes, ", duration, " sec ", size_gb/duration  ," GB/s")

        #print(10**i, " bytes, ", duration, " sec ", runs/duration  ," IOPS")
        measurements.append([10**i, runs/duration])
        #measurements.append(duration)

    print(measurements)
    df = pd.DataFrame(np.asarray(measurements))
    print(df)
    #df.to_csv("/home/ubuntu/object_store_micro/write_objects_iops.csv", header=None)
    return


@ray.remote
def put_objects_2(sz):

    #a = np.ones(sz, dtype=np.uint8)
    d = {}
    for i in range(300):
        k = 'key' + str(i)
        d[k] = np.ones(575000, dtype=np.uint8)

    op = 0
    start = time.time()
    #while (time.time() - start < run_time):
    #runs = int(12*10**9/sz)
    runs=20
    refs=[]
    for j in range(runs):
        #if (j%500==0):
        print(j)
        ref = ray.put(d)
        refs.append(ref)
        # print(ref)
        op+=1
    end = time.time()

    #free(refs)
    sleep(10)

    #print(op)
    return (op, (end - start))

ray.init(address="auto")

res = ray.cluster_resources()
res_keys = res.keys()
nodes = []

for e in res_keys:
    if ('node' in e):
        nodes.append(e)

local_hostname = ray.services.get_node_ip_address()
print(local_hostname)

driver_node_id = f"node:{local_hostname}"

# # Check to make sure the node id resource exists
assert driver_node_id in nodes
nodes.remove(driver_node_id)

# worker_node_id_1 = nodes[0]
# print(driver_node_id, worker_node_id_1)

local_put_func_2 = put_objects_2.options(resources={driver_node_id: 0.01})

#put_objects_2.remote(10**3)

# warm_up = [10**7]
# for i in warm_up:
#     obj_ref = local_put_func_2.remote(i)
#     results = ray.get(obj_ref)
# sleep(30)



print("********************************************************************************************")
#sleep(30)

#obj_sizes=[10**8, 10**8, 10**8, 10**]
workers = [1]

for w in workers:
    for sz in range(5):
        # object_refs = []
        # obj = np.ones(sz,  dtype=np.uint8)
        # for i in range(int(12*10**9/sz)):
        #     object_refs.append(ray.put(obj))

        sleep(10)
        #ray.get(object_refs)
        print("--------------------- obj size: ", sz)
        results = []
        # start = time.time()
        for j in range(1):
            obj_ref = [local_put_func_2.remote(sz) for i in range(w)]
            results = ray.get(obj_ref)
        # end = time.time()

        #free(object_refs)

        op_per_worker = [res[0] for res in results]
        time_per_worker = [res[1] for res in results]

        total_ops = np.sum(np.asarray(op_per_worker))
        max_time = max(time_per_worker)

        lat_per_worker = [res[1]/res[0] for res in results]
        mean_lat = np.median(np.asarray(lat_per_worker))

        print('mean latency is: ', mean_lat*1000, " ms")

        # iops_per_worker = [(res[0]/res[1]) for res in results]
        # average_iops = np.median(np.asarray(iops_per_worker))

        #put_duration = np.median(np.asarray(results[0]))
        #sz_gb = sz/10**9
        #th_per_worker = [sz_gb*i for i in iops_per_worker]
        #average_th = np.median(np.asarray(th_per_worker))

        #system_iops =  total_ops/max_time
        #system_th =  system_iops * sz_gb

        # print("object size: ", sz, "total ops: ", total_ops, "avg throughput (GB/s): ", average_th, "avg iops: ", average_iops)
        # print("system iops: ", system_iops, "system throughput: ", system_th)
        # print("max time: ", max_time)

        measurements.append([sz, mean_lat*1000])
        sleep(30)

# df = pd.DataFrame(data=np.asarray(measurements), columns=['obj_size','Latency(ms)'])
# print(df)
# df.to_csv("/home/ubuntu/object_store_micro/results/write_objects_storage.csv")

