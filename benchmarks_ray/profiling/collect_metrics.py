import numpy as np
import glob
import matplotlib.pyplot as plt
import json
import math

def collect_metrics(worker_files):
    m_dict = {}
    for w in worker_files:
        print(w)
        # iterate and get metrics, add them to dict
        with open(w, "r") as log_file:
            name = w.split("_")[2]
            name = name.split(".")[0]
            if ('driver' in w):
                name = 'driver-' + name
            if ('worker' in w):
                name = 'worker-' + name
            m_dict[name] = [] # change to worker pid
            for line in log_file:
                data = line.split(" ")
                #print(data)
                if ('timestamp:' in data): # metrics available:
                    metrics = []
                    #print(data)
                    i = data.index('timestamp:')
                    metrics.append(data[i+1].replace(',', '')) # timestamp value
                    metrics.append(data[i+4].replace(',', '')) # put requests value
                    metrics.append(data[i+7].replace(',', '')) # get requests value
                    metrics.append(data[i+10].replace(',', '')) # put rate value
                    metrics.append(data[i+13].replace(',', '')) # get rate value
                    metrics.append(data[i+16]) # object sizes value

                    m_dict[name].append(metrics)
    return m_dict

def process_metrics_collected(workers_dict):
    pm_dict={}
    exclude=[]
    workers = workers_dict.keys()
    for w in workers:
        metrics = workers_dict[w]
        pm_dict[w] = {}
        if (len(metrics) == 0):
            continue
        pm_dict[w]['total_put'] = int(metrics[-1][1])
        pm_dict[w]['total_get'] = int(metrics[-1][2])
        pm_dict[w]['put_rate'] = []
        pm_dict[w]['get_rate'] = []
        for l in metrics:
            pm_dict[w]['put_rate'].append(float(l[3]))
            pm_dict[w]['get_rate'].append(float(l[4]))

        pm_dict[w]['total_obs_sz'] = []
        for l in metrics:
            obj = l[5].split("\n")[0]
            if (obj != ""):
                sizes = obj.split(",")
                for sz in sizes:
                    pm_dict[w]['total_obs_sz'].append(int(sz))

        if ((pm_dict[w]['total_put'] == 0) and (pm_dict[w]['total_get'] == 0)):
            if (len(pm_dict[w]['total_obs_sz']) > 0):
                print("Worker ", w, "has no put or get requests, but has an active set of objects with size: ", len(pm_dict[w]['total_obs_sz']))
            pm_dict.pop(w)
            exclude.append(w)
    return pm_dict, exclude

def process_metrics_per_ts(workers_dict, exclude):
    pm_dict={}
    workers = workers_dict.keys()
    for w in workers:
        if (w in exclude):
            continue
        metrics = workers_dict[w]
        pm_dict[w] = []
        if (len(metrics) == 0):
            continue
        for l in metrics:
            if (len(l) != 6):
                continue
            point=[]
            point.append(int(l[0]))
            point.append(int(l[1]))
            point.append(int(l[2]))
            point.append(float(l[3]))
            point.append(float(l[4]))
            objects=l[5].split("\n")[0]
            if (objects==""):
                point.append([])
            else:
                sizes = objects.split(",")
                point.append([int(x) for x in sizes])
            pm_dict[w].append(point)

    return pm_dict


def plot_put_get(workers_dict, plot_put=False, plot_get=False):
    # get timestamp list
    # get put_request list, get_request list
    timestamps={}
    put_list={}
    get_list={}
    points={}
    workers = list(metrics_dict.keys())
    print(workers)

    #workers = ['worker-881', 'worker-7491']
    workers.remove('driver-1850')


    for w in workers:
        #if (w != "worker-20730"):
        #    continue
        timestamps[w]=[]
        put_list[w]=[]
        get_list[w]=[]
        data = workers_dict[w]
        for l in data:
            timestamps[w].append(l[0])
            put_list[w].append(l[1])
            get_list[w].append(l[2])
        start = timestamps[w][0]
        points[w] = [(t-start)/1000 for t in timestamps[w]]

    if (plot_put):
        fig, ax = plt.subplots(figsize=(10,8))
        for w in workers:
            if (w == 'worker-1878'):
                l = 'par server'
            else:
                 l=w
            plt.plot(points[w], put_list[w], linestyle='-', label=l + ' put')
        plt.xlabel('Time(s)', fontsize=14)
        plt.ylabel('Number of requests', fontsize=14)
        plt.legend()
        #plt.show()
        fig.savefig('tf_resnet_gpu/plot_put', bbox_inches = 'tight')

    if (plot_get):
        fig, ax = plt.subplots(figsize=(10,8))
        for w in workers:
            if (w == 'worker-1878'):
                l = 'par server'
            else:
                 l=w
            plt.plot(points[w], get_list[w], linestyle='-', label=l + ' get')
        plt.xlabel('Time(s)', fontsize=14)
        plt.ylabel('Number of requests', fontsize=14)
        plt.legend()
        #plt.show()
        fig.savefig('tf_resnet_gpu/plot_get', bbox_inches = 'tight')


def plot_rate(workers_dict):

    timestamps={}
    put_rate_list={}
    get_rate_list={}
    points={}
    workers = list(workers_dict.keys())
    #workers=['worker-20730']
    #print(workers)

    #workers = ['worker-881', 'worker-7491']
    workers.remove('driver-1850')
    #workers.remove('worker-7491')

    for w in workers:
        timestamps[w]=[]
        put_rate_list[w]=[]
        get_rate_list[w]=[]
        data = workers_dict[w]
        for l in data:
            timestamps[w].append(l[0])
            put_rate_list[w].append(l[3])
            get_rate_list[w].append(l[4])
        start = timestamps[w][0]
        points[w] = [(t-start)/1000 for t in timestamps[w]]

    fig, ax = plt.subplots(figsize=(20,8))
    for w in workers:
        #print(len(points[w]))
        if (w == 'worker-1878'):
             l = 'par server'
        else:
             l=w
        plt.plot(points[w], put_rate_list[w], linestyle='-', label=l + ' put rate')
        plt.xlabel('Time(s)', fontsize=14)
        plt.ylabel('requests/sec', fontsize=14)

    #plt.yscale('log')
    plt.legend(loc=2)
    fig.savefig('tf_resnet_gpu/plot_rate_put', bbox_inches = 'tight')


def plot_hist(data):

    #bins = np.linspace(math.ceil(min(data)),
    #               math.floor(max(data)),
    #               20) # fixed number of bins

    print(min(data), max(data))
    min_lim = math.floor(np.log10(min(data)))
    max_lim = math.floor(np.log10(max(data)))

    print(min_lim, max_lim)

    bins=np.logspace(min_lim,max_lim+1, 50)    #plt.xlim([min(data)-20, max(data)+20])

    plt.hist(data, bins=bins, alpha=0.5, edgecolor='black', linewidth=1.2)

    plt.gca().set_xscale("log")

    plt.xlabel('Object_size(bytes)')
    plt.ylabel('count')
    #print(bins)
    #plt.show()
    plt.savefig('serve/plot_hist.png', bbox_inches = 'tight')

def plot_cdf(data_dict):
    #print(data)
    fig, ax = plt.subplots(figsize=(10,8))

    # bins = np.linspace(math.ceil(min(data)),
    #                math.floor(max(data)),
    #                10)

    workloads = list(data_dict.keys())

    #min_lim = math.floor(np.log10(min(data)))
    #max_lim = math.floor(np.log10(max(data)))

    # print(min_lim, max_lim)

    bins=np.logspace(7,9,200)
    print(workloads)
    for w in workloads:
        data = data_dict[w]
        print(min(data), max(data))

        count, bins_count = np.histogram(data, bins=bins)
        pdf = count / sum(count)
        cdf = np.cumsum(pdf)
        plt.plot(bins_count[1:], cdf, linestyle='--', linewidth=1.5, label=w, alpha=0.8)

    ax.set_xlabel('Object Size(bytes)', fontsize=15)
    ax.set_ylabel('CDF', fontsize=15)

    #plt.xlim(bins[0]-20, bins[-1]+100000)
    plt.gca().set_xscale("log")
    #plt.xticks()
    #plt.show()
    plt.legend(fontsize=13)
    plt.savefig('tf_resnet_gpu/obj_cdf.png', bbox_inches = 'tight')

def total_put_get(metrics_dict):

    workers = metrics_dict.keys()
    total_put=0
    total_get=0
    for w in workers:
        total_put += metrics_dict[w]['total_put']
        total_get += metrics_dict[w]['total_get']
    print("total put: ", total_put, " total get: ", total_get)


def gather_all_data(metrics_dict):
    data = []
    workers = metrics_dict.keys()
    for w in workers:
        data += metrics_dict[w]["total_obs_sz"]
    return data

def store_metrics(filename, workers_dict):
    with open(filename, 'w') as fp:
        json.dump(workers_dict, fp)


def load_dict_json(json_file):
    with open(json_file) as f:
        data = json.load(f)

    return data

def load_mult_json(json_list):
    data = {}
    for json_file in json_list:
        d = load_dict_json(json_file)
        data.update(d)
    return data

#######################################################################################################################
# get metrics here

# path = "/tmp/ray/session_latest/logs"
# worker_files = glob.glob(path+'/python-core-worker-*.log')
# worker_files += glob.glob(path+'/python-core-driver-*.log')

# metrics_dict = collect_metrics(worker_files)
# #print(metrics_dict)
# processed_metrics_dict, exclude_list = process_metrics_collected(metrics_dict)
# print(exclude_list)

# metrics_list = process_metrics_per_ts(metrics_dict, exclude_list)
# #print(metrics_list)
# store_metrics('metrics.json', processed_metrics_dict)
# store_metrics('metrics_list.json', metrics_list)

#######################################################################################################################
# make plots, etc. here

metrics_dict = load_dict_json("tf_resnet_gpu/metrics.json")
metrics_dict_list = load_dict_json("tf_resnet_gpu/metrics_list.json")

# #plot_rate(metrics_dict)
# #print(metrics_dict)
# all_obj_sizes = {}

# metrics_dict = load_mult_json(["serve/metrics.json", "serve/metrics_w1.json", "serve/metrics_w2.json"])
# all_obj_sizes['serve'] = gather_all_data(metrics_dict)

# metrics_dict = load_mult_json(["tf_parserver/metrics.json", "tf_parserver/metrics_w1.json", "tf_parserver/metrics_w2.json"])
# all_obj_sizes['train'] = gather_all_data(metrics_dict)

# metrics_dict = load_mult_json(["rllib_appo/metrics.json", "rllib_appo/metrics_w1.json", "rllib_appo/metrics_w2.json"])
# all_obj_sizes['rl appo'] = gather_all_data(metrics_dict)

# metrics_dict = load_mult_json(["rllib_az/metrics.json", "rllib_az/metrics_w1.json", "rllib_az/metrics_w2.json"])
# all_obj_sizes['rl alphazero'] = gather_all_data(metrics_dict)

# metrics_dict = load_mult_json(["rllib_r2d2/metrics.json", "rllib_r2d2/metrics_w1.json", "rllib_r2d2/metrics_w2.json"])
# all_obj_sizes['rl dqn'] = gather_all_data(metrics_dict)
# plot_cdf(all_obj_sizes)

all_obj_sizes = {}
all_obj_sizes['resnet train'] = gather_all_data(metrics_dict)
#plot_cdf(all_obj_sizes)

# get total put and get
#total_put_get(metrics_dict)

# get/put plots
plot_put_get(metrics_dict_list, plot_put=True)
#
#plot_rate(metrics_dict_list)
