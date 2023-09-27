import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier as dtree


agent="Drone"

har = np.load("../Game/"+agent+"_action_record.npy")
hrr = np.load("../Game/"+agent+"_reward_record.npy")
filename = "../Game/"+agent+"_state_record.pkl"
hsr=pickle.load(open('../Game/'+agent+'_state_record.pkl','rb'))
agents = pickle.load(open('../Game/agents.pkl','rb'))

print(agents)
print(hsr[0])
print(har.shape)
print(hrr.shape)


def action_to_class(action):
  if action[0]==0 and action[1]==0:
    return 0 # none
  if action[0] == 0 and action[1] < 0:
    return 1 # w
  if action[0] > 0 and action[1] < 0:
    return 2 # wd
  if action[0] > 0 and action[1] == 0:
    return 3 # d
  if action[0] > 0 and action[1] > 0:
    return 4 # ws
  if action[0] == 0 and action[1] > 0:
    return 5 # s
  if action[0] < 0 and action[1] > 0:
    return 6 # as
  if action[0] < 0 and action[1] == 0:
    return 7 # a
  if action[0] < 0 and action[1] < 0:
    return 8 # aw
  

print("entering loop")
data = {}

actions = np.array([action_to_class(x) for x in har]) # all the actions for agent a
states = np.array([np.concatenate((h['view'].flatten(),h['object_state'].flatten(),h['memory'].flatten())) for h in hsr]) # all the states agent a saw
rewards = hrr 
print(rewards.shape)

data = { 'actions':actions[rewards!=0], 'states':states[rewards!=0,:] }
print(data['actions'].shape)
print(data['states'].shape)
tree = dtree()
tree.fit(data['states'], data['actions'])
print(tree.predict(data['states']))
print(f"Done Training Tree")

try:
  filehandler = open("./trees/"+agent+"_tree.pkl", 'wb') 
  pickle.dump(tree,filehandler)
except:
  print("crap")

