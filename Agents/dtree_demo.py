import numpy as np
import os
import pickle

fdirs = os.listdir("./trees")

trees={}


for f in fdirs:
  try:
    filehandler = open("./trees/"+f, 'rb') 
    name = f.split("_")[0]
    print(name)
    print(f)
    trees[name]=(pickle.load(filehandler))
  except:
    print("crap") 

def class_to_action(action):
  if action == 0: # none
    return np.array([0,-1])
  if action == 1: # w
    return np.array([0,-1])
  if action == 2: # wd
    return np.array([1,-1])
  if action == 3: # d
    return np.array([1,0])
  if action == 4: # ds
    return np.array([1,1])
  if action == 5: # s
    return np.array([0,1])
  if action == 6: # sa
    return np.array([-1,1])
  if action == 7: # a
    return np.array([-1,0])
  if action == 8: # aw
    return np.array([-1,-1])


# Time to play the game with these trees
import sys
sys.path.insert(1, '../Game') # lets us import game from another folder
from game import sar_env
import pickle

agents = ["Human","RoboDog","Drone"]
pois = ["Child", "Child", "Adult"]
premade_map = np.load("../LevelGen/Island/Map.npy")
game = sar_env(display=True, tile_map=premade_map, agent_names=agents, poi_names=pois)
state, info = game.start()
terminated = False

while not terminated:
  actions = np.zeros((len(agents),2))
  messages = np.zeros((len(agents),2))
  for i,a in enumerate(agents):
    tree_state = [np.concatenate((state['view'][i].flatten(),state['object_state'][i].flatten(),state['memory'][i].flatten()))]
    #print(class_to_action(trees[a].predict(tree_state)))
    actions[i] = class_to_action(trees[a].predict(tree_state)) # gets the tree's prediction class back to an action
  state, rewards, terminated, truncated, info = game.step(actions=actions, messages=messages)
