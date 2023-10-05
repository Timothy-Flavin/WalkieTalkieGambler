from brain import brain
from halsarc.Game.game import sar_env
import os
import numpy as np
# define policy network

class boid(brain):
  def __init__(self, sar, a_type=0):
    super(boid,self).__init__('boid',0)
    self.sar = sar
  def com(self, a, norm=False, invsq=False):
    rc = int(a.shape[0]/2)
    cc = int(a.shape[1]/2)
    xcom = 0
    ycom = 0
    for r in range(a.shape[0]):
      for c in range(a.shape[1]):
        dy=a[r,c]*(r-rc)
        dx=a[r,c]*(c-cc)
        if invsq and r!=rc and c!=cc:
          d = (r-rc)*(r-rc)+(c-cc)*(c-cc)
          dy/=d
          dx/=d
        xcom+=dx
        ycom+=dy
    com = np.array([xcom,ycom])
    if norm and np.sum(np.square(com))>0:
      com/=np.sqrt(np.sum(np.square(com)))
    return com
  
  def get_poi_dir(self,state, anum):
    pos = state['object_state'][anum]['a_state'][0,0:2]
    dst = 10000000
    p_index = -1
    dir = np.zeros(2)
    #print(f"pstate: \n{state['object_state'][anum]['p_state']}")
    for i in range(state['object_state'][anum]['p_state'].shape[0]):
      #poi numbers: x,y,destroyed,saved,age,recency
      poi = state['object_state'][anum]['p_state'][i]
      #print(f"poi: {poi}, pos: {pos}")
      if np.sum(np.square(poi[0:2]))>0 and poi[2]>0 and poi[3]<1 \
        and np.sum(np.square(poi[0:2]-pos)) < dst and poi[6]:
        dst = np.sum(np.square(poi[0:2]-pos))
        p_index = i
    #print(f"{anum}, poi {p_index}")
    if p_index > -1:
      print(self.sar.pois[p_index].saved)
      print(f"anum: {anum},\n  {pos} \n  pstate: {state['object_state'][anum]['p_state'][p_index]}")
      dir = state['object_state'][anum]['p_state'][p_index,0:2]-pos
      #print(f"before norm: {dir}")
      if np.sum(np.square(dir)) > 0:
        dir = dir/np.sqrt(np.sum(np.square(dir)))
      print(f"after norm: {dir}")
    return dir
  
  def action(self,state,anum):
    #print("Starting action")
    speed_com = self.com(state['view'][anum,0], norm=True)
    alt_com = self.com(state['view'][anum,1], norm=True,invsq=True)
    memory_com = self.com(state['view'][anum,3], norm=True,invsq=True)
    trail_com = self.com(state['view'][anum,2], norm=True,invsq=True)
    rand_com = np.random.random(2)-0.5
    #print(state['object_state'][anum]['a_state'])
    center_com = -state['object_state'][anum]['a_state'][0,0:2] 
    #print(f"anume: {anum}'s pstate: {state['object_state'][anum]['p_state']}")
    #print(center_com)
    poi_dir = state['object_state'][anum]['p_state'][0:2]+center_com
    poi_mag = state['object_state'][anum]['p_state'][2]
    #print(state['view'][2])
    #print(state['view'][anum,0])
    #print(speed_com)
    #print(alt_com)
    #print(memory_com)
    #input()

    action = np.zeros((1,14+self.sar.max_agents))
    action[0,0:2] = speed_com+alt_com-memory_com+rand_com+center_com+poi_dir*5*poi_mag+trail_com
    #print(action.shape)
    for i in range(5):
      if state['radio']["message_legality"][anum][i]>0:
        action[0,2] = 1
        action[0,3] = action[0,0]
        action[0,4] = action[0,1]
        action[0,5] = 3
        action[0,6+i] = 1
        break
    return action
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
    
if __name__ == "__main__":
  print()