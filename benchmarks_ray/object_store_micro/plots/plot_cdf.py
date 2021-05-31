
# defining the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# No of Data points
local_data = np.loadtxt('../results/read_local_latency.csv')
remote_data = np.loadtxt('../results/read_remote_latency.csv')[:-1]
print(local_data[0])

local_data = local_data*1000
remote_data = remote_data*1000

colors = ['#6187a5', '#896689', '#cc7e81', '#f4a680', '#cec591']

N = local_data.shape[0]
bins = np.arange(local_data[0], remote_data[-1], step=50)
#print(bins)
# getting data of the histogram
local_count, local_bins_count = np.histogram(local_data, bins=bins)
# # finding the PDF of the histogram using count values
local_pdf = local_count / sum(local_count)
# # using numpy np.cumsum to calculate the CDF
# # We can also find using the PDF values by looping and adding
local_cdf = np.cumsum(local_pdf)

# getting data of the histogram
remote_count, remote_bins_count = np.histogram(remote_data, bins=bins)
# # finding the PDF of the histogram using count values
remote_pdf = remote_count / sum(remote_count)
# # using numpy np.cumsum to calculate the CDF
# # We can also find using the PDF values by looping and adding
remote_cdf = np.cumsum(remote_pdf)

# # plotting PDF and CDF
#plt.plot(bins_count[1:], pdf, color="red", label="PDF")

fig, ax = plt.subplots(figsize=(10,8))
labels = ['local', 'remote']

plt.plot(local_bins_count[1:], local_cdf, linestyle='-', linewidth=2, label="local", color=colors[3])
plt.plot(remote_bins_count[1:], remote_cdf, linestyle='-', linewidth=2, label="remote", color=colors[1])
plt.xlim(0.1)

print('95th: ', np.percentile(local_data, 95))
print('95th: ', np.percentile(remote_data, 95))

ax.set_xlabel('Read latency(us)', fontsize=14)
ax.set_ylabel('CDF', fontsize=14)
ax.set_title('Read latency CDF, 10KB object')
plt.legend()
fig.savefig('latency/cdf.png', bbox_inches = 'tight')
