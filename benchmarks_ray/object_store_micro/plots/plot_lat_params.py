import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

colors = ['#6187a5', '#896689', '#f4a680', '#cec591']

df = pd.read_csv('../results/remote_latency.csv')

#threads=2
chunk_size=0.1
obj_size=100

chunk_sizes = []
times = []
threads=[]

for index,row in df.iterrows():
    # if (row['Object size(MB)'] == obj_size and row['Threads'] == threads):
    #      chunk_sizes.append(row['Chunk Size(MB)'])
    #      times.append(row['Time(ms)'])
    if (row['Object size(MB)'] == obj_size and row['Chunk Size(MB)'] == chunk_size):
        threads.append(row['Threads'])
        times.append(row['Time(ms)'])

print(threads, times)

x_values = np.arange(len(threads))
y_values = times

x_labels = ['1', '2', '4', '8']
#x_labels = ['100KB', '1MB', '5MB', '10MB']

fig, ax = plt.subplots(figsize=(10,8))
ax.bar(x_values, y_values, width=0.4, color=colors[2])
#lns+=line

plt.xticks(x_values, x_labels)


ax.set_xlabel('chunk size', fontsize=14)
ax.set_ylabel('time(ms)', fontsize=14)
ax.set_title('Numpy array 100MB, fetch from remote memory, 2 rpc threads')


#ax.legend(lns, labs, fontsize=12)
#plt.legend()

#fig.savefig('latency/read_latency_remote_chunk_size.png', bbox_inches = 'tight')

plt.show()