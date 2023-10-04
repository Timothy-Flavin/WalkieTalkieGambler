import gymnasium as gym
from swig import *
from TD3_Brain import td3_brain
import numpy as np
import matplotlib.pyplot as plt
env = gym.make("MountainCarContinuous-v0")
print(env.action_space.shape)
print(env.observation_space.shape)
observation, info = env.reset()
class sa():
    def __init__(self,ma):
        self.max_agents=ma

def vec(a,b,c):
    return a

brain = td3_brain(0,"drone",sa(-13),observation,vec,32,"./sanity", action_dim=1,max_act=2)
ep_r = 0
tot_r = []

nep = 0

while nep < 500:
    action = brain.action(observation,0)[0] # agent policy that uses the observation and info
    #print(action)
    observation_, reward, terminated, truncated, info = env.step(action)
    ep_r+=reward
    brain.update(0,observation,action,reward,observation_,terminated,truncated,None)
    observation = observation_
    if terminated or truncated:
        print(nep)
        nep+=1
        tot_r.append(ep_r)
        ep_r=0
        observation, info = env.reset()
        


env = gym.make("MountainCarContinuous-v0",render_mode = 'human')
observation, info = env.reset()
end=False
while not end:
    action = brain.action(observation,0)[0] # agent policy that uses the observation and info
    #print(action)
    observation_, reward, terminated, truncated, info = env.step(action)
    ep_r+=reward
    brain.update(0,observation,action,reward,observation_,terminated,truncated,None)
    observation = observation_
    if terminated or truncated:
        nep+=1
        tot_r.append(ep_r)
        ep_r=0
        observation, info = env.reset()
        
        plt.plot(tot_r)
        plt.show()


env.close()