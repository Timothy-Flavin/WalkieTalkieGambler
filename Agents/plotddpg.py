import numpy as np
import matplotlib.pyplot as plt

fnames = ['boid','td3_brain','ppo_boid','ppo_brain','td3_big_brain','ppo_big_brain']
anames = ['Human','RoboDog','Drone']
npars = {}
files = {}
names= {}
#colors = ['green','blue','orange','red','purple','yellow']
for a in anames:
  files[a] = []
  npars[a] = []
  names[a] = []
  for f in fnames:
    files[a].append(f"./{f}/{a}/rewards.npy")
    npars[a].append(np.load(files[a][-1]))
    names[a].append(f"({f})")
    print(npars[a][-1].shape)
print(files)

factor = 20
smooth_arrs = {}
for a in anames:
  smooth_arrs[a] = []
  for arr in npars[a]:
    smooth_arrs[a].append([])
    for i in range(int(len(arr)/factor*2)):
      smooth_arrs[a][-1].append(np.mean(arr[int(i/2*factor):int((i+1)*factor/2)]))
    print(len(smooth_arrs[a][-1]))
  #print(len(smooth_arrs[a]))

for a in anames:
  for i,s in enumerate(smooth_arrs[a]):
    plt.plot(s,label=names[a][i],linewidth=1.5)
  plt.legend()
  plt.grid()
  plt.title(f"Results for agent {a}")
  plt.show()

