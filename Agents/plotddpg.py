import numpy as np
import matplotlib.pyplot as plt

fnames = ["ddpg","rand_agent","torch_sup",'boid']
anames = ['Human','RoboDog','Drone']
npars = []
files = []
names=[]
colors = ['green','blue','red','orange']
for f in fnames:
  for a in anames:
    files.append(f"./{f}/{a}/rewards.npy")
    npars.append(np.load(files[-1]))
    names.append(f"{a}({f})")
    print(npars[-1].shape)
print(files)

factor = 25
smooth_arrs = []
for arr in npars:
  smooth_arrs.append([])
  for i in range(int(len(arr)/factor)):
    smooth_arrs[-1].append(np.mean(arr[i*factor:(i+1)*factor]))
  print(len(smooth_arrs[-1]))
print(len(smooth_arrs))

for i,s in enumerate(smooth_arrs):
  plt.plot(s,label=names[i],linewidth=1.5,color=colors[int(i/len(anames))])
  #plt.show()

plt.legend()
plt.grid()
plt.show()