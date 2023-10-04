from brain import brain
from halsarc.Game.game import sar_env
import os
import numpy as np
from ppo_git import PPO
# define policy network

class ppo_brain(brain):
  def __init__(self, anum,a, sar, state, vectorizer = sar_env.vectorize_state, update_every=32, fname='ppo_brain'):
    super(ppo_brain,self).__init__('ppo_brain',anum)
    self.fname= fname
    self.update_every = update_every
    self.frame = 0
    self.num_ac = 0
    self.num_up = 0
    self.a_type = a
    self.sar = sar
    self.vectorizer = vectorizer
    nn_state = self.vectorizer(state,anum,True)
    self.ppo = PPO(state_dim=nn_state.shape[0],
                    action_dim=2,
                    lr_actor=0.0005,
                    lr_critic=0.001,
                    gamma=0.997,
                    K_epochs=100,
                    eps_clip=0.005,
                    has_continuous_action_space=True,
                    action_std_init=0.6)

  def action(self,state,anum):
    self.num_ac+=1
    act = np.zeros((1,14+self.sar.max_agents))
    act[0,0:2] = self.ppo.select_action(self.vectorizer(state,anum,True))
    return act
  # This is where a model will be trained if in training mode.
  def update(self,anum,state,action,rewards,state_,terminated,truncated,game_instance):
    self.frame+=1
    self.num_up+=1
    self.ppo.buffer.rewards.append(rewards)
    self.ppo.buffer.is_terminals.append(float(int(terminated or truncated)))
        
    if self.frame>=self.update_every:
      #print(f"num ac {self.num_ac}, + num up: {self.num_up}")
      self.ppo.update()
      self.ppo.decay_action_std(0.0005, 0.1)
      self.frame=0
  # for algorithms that can only update after an episode
  def update_end_of_episode(self):
    pass
  # This will be called every so many minutes to save the model in case
  # of crash or other problems
  def checkpoint(self):
    self.ppo.save(f"./{self.fname}/{self.a_type}/policy")
  # load this model from a checkpoint
  def load(self):
    self.ppo.load(f"./{self.fname}/{self.a_type}/policy")
    
if __name__ == "__main__":
  print()