import numpy as np
import matplotlib.pyplot as plt

fnames = ['boid','sac_boid','sac_brain','sac_big_brain','rand_agent']#'ppo_boid','ppo_brain','ppo_big_brain'

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
    print(files[a][-1])
    print(npars[a][-1].shape)
print(files)

factor = 50
smooth_arrs = {}
for a in anames:
  smooth_arrs[a] = []
  for arr in npars[a]:
    smooth_arrs[a].append([])
    for i in range(len(arr)-factor):
      smooth_arrs[a][-1].append(np.mean(arr[i:i+factor]))
    print(len(smooth_arrs[a][-1]))
  #print(len(smooth_arrs[a]))

for a in anames:
  for i,s in enumerate(smooth_arrs[a]):
    #print(f"a {a}, i {names[a][i]}, s {s}")
    
    plt.plot(s,label=names[a][i],linewidth=1.5)
  plt.legend()
  plt.grid()
  plt.title(f"Results for agent {a}")
  plt.show()

