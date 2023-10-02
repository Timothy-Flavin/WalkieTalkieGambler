from brain import brain
from halsarc.Game.game import sar_env
import os
import numpy as np
# define policy network

class rand_agent(brain):
  def __init__(self, sar):
    super(rand_agent,self).__init__('random',0)
    self.sar = sar
  def action(self,state,anum):
    return self.sar.random_action()
  # This is where a model will be trained if in training mode.
  def update(self,anum,state,action,rewards,state_,terminated,truncated,game_instance):
    pass
  # for algorithms that can only update after an episode
  def update_end_of_episode(self):
    pass
  # This will be called every so many minutes to save the model in case
  # of crash or other problems
  def checkpoint(self):
    pass
  # load this model from a checkpoint
  def load(self):
    pass
    
