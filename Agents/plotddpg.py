import numpy as np
import matplotlib.pyplot as plt

fnames = ["ddpg","ddpg_1_6","ddpg_1_4","ddpg_1_2","ppo"]
npars = []
files = []
for f in fnames:
  files.append(f"./{f}/rewards.npy")
  npars.append(np.load(files[-1]))

factor = 100
smooth_arrs = []
for arr in npars:
  smooth_arrs.append([])
  for i in range(int(len(arr)/factor)-1):
    smooth_arrs[-1].append(np.mean(arr[i*factor:(i+1)*factor]))

for i,s in enumerate(smooth_arrs):
  plt.plot(s,label=fnames[i])

plt.legend()
plt.grid()
plt.show()