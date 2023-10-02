import numpy as np
import random
from halsarc.Game.game import sar_env
from torch_sup import t_sup, sup_policy
from ddpg import ddpg, policy_net, value_net
from rand_agent import rand_agent
import os
import time
from boid import boid

def load_models():
  brains = {}
  end_rewards = {}
  brains['ddpg'] = {}
  end_rewards['ddpg'] = {}
  for a in agents:
    brains['ddpg'][a] = ddpg(anum=0,
                        state_size=st.shape[0],
                        max_agents=3,
                        m_types=8,
                        max_instruction=5,
                        hidden_dims=[128,128],
                        tau=0.95,
                        directory=f"./ddpg/{a}/",
                        batch_size=32,
                        update_every=64,
                        eps=eps,
                        eps_decay=0.9990)
    if not os.path.exists(f"./ddpg/{a}/"):
      os.mkdir(f"./ddpg/{a}/")
    try:
      end_rewards['ddpg'][a] = np.load(f"./ddpg/{a}/rewards.npy").tolist()
    except: 
      end_rewards['ddpg'][a] = []
      print(f"Could not find rewards at: ./ddpg/{a}/rewards.npy")
  
  
  brains['torch_sup'] = {}
  end_rewards['torch_sup'] = {}
  for a in agents:
    brains['torch_sup'][a] = t_sup(anum = -1,
                        state_size=st.shape[0],
                        data_dir=f"./recorded_data/{a}",
                        directory=f"./torch_sup/{a}/")
    if not os.path.exists(f"./torch_sup/{a}/"):
      os.mkdir(f"./torch_sup/{a}/")
    try:
      end_rewards['torch_sup'][a] = np.load(f"./torch_sup/{a}/rewards.npy").tolist()
    except: 
      end_rewards['torch_sup'][a] = []
      print(f"Could not find rewards at: ./torch_sup/{a}/rewards.npy")
  
  
  brains['rand_agent'] = {}
  end_rewards['rand_agent'] = {}
  for a in agents:
    brains['rand_agent'][a] = rand_agent(env)
    if not os.path.exists(f"./rand_agent/{a}/"):
      os.mkdir(f"./rand_agent/{a}/")
    try:
      end_rewards['rand_agent'][a] = np.load(f"./rand_agent/{a}/rewards.npy").tolist()
    except: 
      end_rewards['rand_agent'][a] = []
      print(f"Could not find rewards at: ./rand_agent/{a}/rewards.npy")

  brains['boid'] = {}
  end_rewards['boid'] = {}
  for a in agents:
    brains['boid'][a] = boid(env)
    if not os.path.exists(f"./boid/{a}/"):
      os.mkdir(f"./boid/{a}/")
    try:
      end_rewards['boid'][a] = np.load(f"./boid/{a}/rewards.npy").tolist()
    except: 
      end_rewards['boid'][a] = []
      print(f"Could not find rewards at: ./boid/{a}/rewards.npy")
  return brains, end_rewards

if __name__ == "__main__":
  player_num = -1
  agents = ["Human","RoboDog","Drone"]
  max_agents = len(agents)
  pois = ["Child", "Child", "Adult"]
  premade_map = np.load("../LevelGen/Island/Map.npy")
  env = sar_env(max_agents=3,display=True, tile_map=premade_map, agent_names=agents, poi_names=pois,seed=random.randint(0,10000),player=player_num,explore_multiplier=0.005)
  state, info = env.start()
  st = sar_env.vectorize_state(state,0,True)
  print(f"Message shape: {state['radio']['message'].shape}")
  print(f"State shape: {st.shape}")

  eps = 0.45
  terminated = False
  # instantiate the policy
  brain_names = ['boid']#'ddpg','torch_sup','rand_agent',
  brains, end_rewards = load_models()
  # create an optimizer
  # initialize gamma and stats
  n_episode = -1
  render_rate = 50 # render every render_rate episodes
  
  while True:
    first_start = time.time()
    n_episode+=1
    player_num+=1
    player_num = player_num%len(agents)
    env.player = player_num
    selected_brains = []
    for i in range(max_agents):
      selected_brains.append(brain_names[random.randint(0,len(brain_names)-1)])
    if n_episode%render_rate==0:
      #plt.plot(envrew)
      #plt.title("rewards over time")
      #plt.show()
      
      env.display=True
    else:
      env.display=False
    tot_r = np.zeros(3)
    # reset environment
    state, info = env.start()
    for i in range(len(agents)):
      env.agents[i].brain_name = selected_brains[i]
    while True:
      # render episode every render_rate epsiodes
      # calculate probabilities of taking each action
      #print(state['object_state'])
      np_actions = []
      for i,a in enumerate(agents):
        brains[selected_brains[i]][a].dead = info.agents[i].destroyed
        if brains[selected_brains[i]][a].dead:
          np_actions.append(np.zeros((1,14+max_agents)))
        else:
          np_actions.append(brains[selected_brains[i]][a].action(state,i))
          #print(f"act shape: {np_actions[-1].shape}")
      np_actions = np.array(np_actions)
      np_actions = np_actions.reshape((np_actions.shape[0],np_actions.shape[2]))
      
      new_state, reward, done, _, info = env.step(np_actions)
      #print(new_state['view'][0])
      #input()
      tot_r += reward
      new_nn_state=[]
      for i,a in enumerate(agents):
        new_nn_state.append(
          sar_env.vectorize_state(new_state,i,True)
        )# store state, action and reward
        #print(f"new state shape: {new_nn_state[0].shape}")
        brains[selected_brains[i]][a].update(i,state,np_actions[i],reward[i],new_state,done,_,env)
      state = new_state
      if done:
        break
    print(f"eps: {brains['ddpg'][agents[0]].eps}")
    for i,a in enumerate(agents):
      end_rewards[selected_brains[i]][a].append(tot_r[i])
      print(f"{a}({selected_brains[i]}) r: {tot_r[i]}")
    for a in agents:
      for b in brain_names:
        brains[b][a].checkpoint()
        np.save(f"./{b}/{a}/rewards.npy",np.array(end_rewards[b][a]))