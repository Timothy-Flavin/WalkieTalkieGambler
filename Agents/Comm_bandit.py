import numpy as np
import torch
from torch.distributions import Categorical

dist = Categorical(torch.from_numpy(np.array([0.0,0.3,0.1,0.2])))
s = dist.sample()
slp = dist.log_prob(s)

print(s)
print(slp)
    #long