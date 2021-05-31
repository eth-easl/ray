import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

colors = ['#6187a5', '#896689', '#cc7e81', '#f4a680', '#cec591']

def get_files(input_files):
    df_list = []
    for f in input_files:
        path = '..//results/' + f
        df_list.append(pd.read_csv(path))
    return df_list


df_list = get_files(['read_objects_remote_np.csv', 'read_objects_storage_np.csv'])
#df_list = get_files(['write_objects_1_worker.csv'])

fig, ax = plt.subplots(figsize=(10,8))
x_labels = ['100KB', '1MB', '10MB', '100MB']
labels = ['remote Object Store', 'local SSD']

labs=[]
lns=[]

for i in range(len(labels)):
    print(df_list[i])
    x_values = np.asarray(df_list[i]['obj_size'])
    y_values = np.asarray(df_list[i]['time'])
    if (i==1):
        line = plt.plot(x_values[1:], y_values[1:], marker='o', linestyle='-', color=colors[i], label=labels[i])
    else:
        line = plt.plot(x_values[2:], y_values[2:], marker='o', linestyle='-', color=colors[i], label=labels[i])
    lns += line

labs = [l.get_label() for l in lns]

ax.set_xscale('log')

ax.set_xticks(np.asarray(df_list[0]['obj_size'][2:]))
ax.set_xticklabels(x_labels)

ax.set_yscale('log')

ax.set_xlabel('object size', fontsize=14)
ax.set_ylabel('time(ms)', fontsize=14)
ax.legend(lns, labs, fontsize=12)

ax.set_title('Read latency, 1 worker, numpy array', fontsize=14)
fig.savefig('latency/read_latency_storage_remote.png', bbox_inches = 'tight')

