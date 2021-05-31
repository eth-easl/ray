import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

colors = ['#6187a5', '#896689', '#f4a680', '#cec591']

df = pd.read_csv('../results/breakdown/breakdown_storage_np.csv')
print(df)
num_workloads = 3

#labels = ['Deserialization', 'Plasma', 'Raylet']
labels = ['Deserialization', 'Restore']
#labels = ['Get from Plasma','Communication and transferring','Put into Plasma','Deserialization']
#data = [list(df['Deser(ms)']), list(df['Plasma(ms)']),list(df['Raylet(ms)'])]
data = [list(df['Deser(ms)']), list(df['Restore(ms)'])]
#data = [d[1:] for d in data]
#data = [list(df['Prep(ms)']), list(df['Comm_and_transfer(ms)']),list(df['Write(ms)']), list(df['Deser(ms)'])]
data = [d[:-2] for d in data]

print(data)
fig, ax = plt.subplots(figsize=(10,8))
x = np.arange(num_workloads)
x_labels = ['100KB', '1MB', '5MB']
barWidth = 0.4

print(data)
ax.bar(x, data[0], barWidth,label=labels[0], color='darkgreen')
ax.bar(x, data[1], barWidth, bottom=data[0], label=labels[1], color=colors[1])
#ax.bar(x, data[2], barWidth, bottom=np.array(data[0])+np.array(data[1]), label=labels[2], color=colors[2])
#ax.bar(x, data[3], barWidth, bottom=np.array(data[0])+np.array(data[1])+np.array(data[2]), label=labels[3], color=colors[3])


ax.set_ylabel('Time(ms)', fontsize=12)
ax.set_xlabel('Object Size', fontsize=12)
ax.set_title('Numpy array from storage, access breakdown')

plt.xticks(x, x_labels)
#ax.set_yscale("log")

ax.legend(fontsize=12)

plt.savefig("breakdown/breakdown_storage_np_small.png",  bbox_inches = 'tight')
#plt.show()

