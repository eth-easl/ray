import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

colors = ['#6187a5', '#896689', '#f4a680', '#cec591']

df = pd.read_csv('slowdown_ips.csv')
print(df)
num_workloads = 3
lat = list(df['latency'])[1:]
data = [list(df['resnet_gpu_slowdown'])[1:]]

print(data)
#labels = ['serve', 'train', 'rl appo', 'rl r2d2']

fig, ax = plt.subplots(figsize=(10,8))
x = np.arange(num_workloads)
x_labels = ['100us', '1ms', '10ms']
barWidth = 0.3

for i in range(1):
    line=plt.bar(x+barWidth*i, data[i], width=barWidth, color=colors[1], )

plt.xlabel('Added latency', fontsize = 15)
plt.ylabel('Slowdown(%)', fontsize = 15)
plt.xticks(x, x_labels)
plt.title("Resnet-101 training, 4 GPUs", fontsize=14)

#plt.legend()
#plt.show()
plt.savefig("slowdown_gpu_resnet.png",  bbox_inches = 'tight')

# num_workloads = 5
# x =  np.arange(num_workloads)
# rem_perc = [30, 48, 30, 10, 18]
# bytes_transf = [200, 30, 60, 0.02, 36]
# cpu_util = [50, 49.6, 43, 80, 60]
# mem_util = [25.3, 14.3, 15, 15, 13]
# labels = ['train', 'serve', 'rl appo', 'rl alphazero', 'rl dqn']
# fig = plt.figure(figsize = (7, 5))

# plt.bar(x, mem_util, color =colors[3], width=0.4)

# plt.xlabel("Workload", fontsize=11)
# #plt.ylabel("Percentage(%) of remote accesses", fontsize=11)
# # plt.ylabel("GB transferred between nodes", fontsize=11)
# # plt.yscale('log')
# # plt.yticks([0.01, 0.1, 1, 10, 100], ['0.01', '0.1', '1', '10', '100'])

# plt.ylabel("Mem utilization(%)", fontsize=11)


# #plt.title("")
# plt.xticks([r for r in range(num_workloads)], labels, fontsize=11)
# plt.savefig("mem_util.png",  bbox_inches = 'tight')