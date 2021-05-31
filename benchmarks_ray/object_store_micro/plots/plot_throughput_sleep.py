import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

colors = ['#6187a5', '#896689', '#cc7e81', '#f4a680', '#cec591']
header = ['workers', 'obj_size','System IOPS', 'System Throughput(GB/s)', 'Worker IOPS', 'Worker Throughput(GB/s)', 'Latency(ms)', 'Avg Ops']

path_local = '../results/read_objects_throughput_local_new2.csv'
path_remote = '../results/read_objects_throughput_remote_new2.csv'

df_local = pd.read_csv(path_local)
df_remote = pd.read_csv(path_remote)

workers = df_local['workers'][0:-1]
local_iops = df_local['System IOPS'][0:-1]
remote_iops = df_remote['System IOPS'][0:-1]

#print(sleep_time)
# obj_size = df['obj_size']
# iops = df['System IOPS']
# throughput = df['System Throughput(GB/s)']
# latency = df['Latency(ms)']

#x_labels = ['1KB', '10KB', '100KB', '1MB', '10MB']
x_labels = [str(int(i)) for i in workers]
#labels = ['memcpy - 2 threads', 'memcpy - 6 threads', 'memcpy - 8 threads']

x_range = list(range(len(workers)))

fig, ax = plt.subplots(figsize=(10,8))

labs=[]
lns=[]

# ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
# ax2.set_ylabel('Latency(ms)')  # we already handled the x-label with ax1
# line=ax2.bar(x_range, latency, width=0.5, color=colors[1], label='latency', zorder=0, alpha=0.6)
# lns+=line

line=ax.plot(x_range, local_iops, marker='o', linestyle='-', color=colors[0], label='local', zorder=1)
lns+=line

line=ax.plot(x_range, remote_iops, marker='o', linestyle='-', color=colors[2], label='remote', zorder=1)
lns+=line

labs = [l.get_label() for l in lns]

#print(x_labels)

ax.set_xticks(x_range)
ax.set_xticklabels(x_labels)
ax.set_xlabel('number of clients', fontsize=14)
ax.set_ylabel('IOPS', fontsize=14)

ax.legend(loc=0, fontsize=10)
# ax2.legend(loc=1, fontsize=10)


ax.set_title('Read operation, 64KB, 16 clients', fontsize=14)
#plt.yscale("log", base=2)
fig.savefig('throughput/read_iops_local_remote_new.png', bbox_inches = 'tight')
#plt.show()
