import numpy as np
import random
import torch
from halsarc.Game.game import sar_env
from torch_sup import t_sup, sup_policy
from ddpg import ddpg, policy_net, value_net
from rand_agent import rand_agent
import os
import time
from boid import boid
from ppo_agent import ppo_brain
from TD3_Brain import td3_brain
from SAC_brain import SAC_Brain 
from SAC_radio import SAC_radio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Each model implements the brain.py interface but they have
# different hyper parameters so this function loads all the 
# required models in by name
def load_models(state, agents, env,brain_names):
  brains = {}
  end_rewards = {}


  if 'ddpg' in brain_names:
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
  
  if 'torch_sup' in brain_names:
    tsupst = sar_env.vectorize_state_small(state,0,True)
    brains['torch_sup'] = {}
    end_rewards['torch_sup'] = {}
    for a in agents:
      brains['torch_sup'][a] = t_sup(anum = -1,
                          state_size=tsupst.shape[0],
                          data_dir=f"./recorded_data/{a}",
                          directory=f"./torch_sup/{a}/")
      if not os.path.exists(f"./torch_sup/{a}/"):
        os.mkdir(f"./torch_sup/{a}/")
      try:
        end_rewards['torch_sup'][a] = np.load(f"./torch_sup/{a}/rewards.npy").tolist()
      except: 
        end_rewards['torch_sup'][a] = []
        print(f"Could not find rewards at: ./torch_sup/{a}/rewards.npy")
    
  if 'rand_agent' in brain_names:
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

  if 'boid' in brain_names:
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
  
  if 'ppo_brain' in brain_names:
    brains['ppo_brain'] = {}
    end_rewards['ppo_brain'] = {}
    for a in agents:
      brains['ppo_brain'][a] = ppo_brain(-1,a,env,state,sar_env.vectorize_state_small,64)
      if not os.path.exists(f"./ppo_brain/{a}/"):
        os.makedirs(f"./ppo_brain/{a}/")
      try:
        end_rewards['ppo_brain'][a] = np.load(f"./ppo_brain/{a}/rewards.npy").tolist()
      except: 
        end_rewards['ppo_brain'][a] = []
        print(f"Could not find rewards at: ./ppo_brain/{a}/rewards.npy")
  
  if 'ppo_boid' in brain_names:
    brains['ppo_boid'] = {}
    end_rewards['ppo_boid'] = {}
    for a in agents:
      brains['ppo_boid'][a] = ppo_brain(-1,a,env,state,sar_env.boid_state,fname='ppo_boid')
      if not os.path.exists(f"./ppo_boid/{a}/"):
        os.makedirs(f"./ppo_boid/{a}/")
      try:
        end_rewards['ppo_boid'][a] = np.load(f"./ppo_boid/{a}/rewards.npy").tolist()
      except: 
        end_rewards['ppo_boid'][a] = []
        print(f"Could not find rewards at: ./ppo_boid/{a}/rewards.npy")

  if 'td3_boid' in brain_names:
    brains['td3_boid'] = {}
    end_rewards['td3_boid'] = {}
    for a in agents:
      brains['td3_boid'][a] = td3_brain(0,a,env,state,sar_env.boid_state,1,'td3_boid',eps=0,eps_decay=1,update_after=1000)
      if not os.path.exists(f"./td3_boid/{a}/"):
        os.makedirs(f"./td3_boid/{a}/")
      try:
        end_rewards['td3_boid'][a] = np.load(f"./td3_boid/{a}/rewards.npy").tolist()
      except: 
        end_rewards['td3_boid'][a] = []
        print(f"Could not find rewards at: ./td3_boid/{a}/rewards.npy")

  if 'td3_brain' in brain_names:
    fname = 'td3_brain'
    brains[fname] = {}
    end_rewards[fname] = {}
    for a in agents:
      brains[fname][a] = td3_brain(0,a,env,state,sar_env.vectorize_state_small,1,'td3_boid',eps=0,eps_decay=1,update_after=1000)
      if not os.path.exists(f"./{fname}/{a}/"):
        os.makedirs(f"./{fname}/{a}/")
      try:
        end_rewards[fname][a] = np.load(f"./{fname}/{a}/rewards.npy").tolist()
      except: 
        end_rewards[fname][a] = []
        print(f"Could not find rewards at: ./{fname}/{a}/rewards.npy")
  
  if 'ppo_big_brain' in brain_names:
    fname = 'ppo_big_brain'
    brains[fname] = {}
    end_rewards[fname] = {}
    for a in agents:
      brains[fname][a] = ppo_brain(-1,a,env,state,sar_env.boid_state,fname=fname)
      if not os.path.exists(f"./{fname}/{a}/"):
        os.makedirs(f"./{fname}/{a}/")
      try:
        end_rewards[fname][a] = np.load(f"./{fname}/{a}/rewards.npy").tolist()
      except: 
        end_rewards[fname][a] = []
        print(f"Could not find rewards at: ./{fname}/{a}/rewards.npy")

  if 'td3_big_brain' in brain_names:
    fname = 'td3_big_brain'
    brains[fname] = {}
    end_rewards[fname] = {}
    for a in agents:
      brains[fname][a] = td3_brain(0,a,env,state,sar_env.vectorize_state,1,'td3_boid',eps=0,eps_decay=1,update_after=1000)
      if not os.path.exists(f"./{fname}/{a}/"): # eps .1 update after 25k
        os.makedirs(f"./{fname}/{a}/")
      try:
        end_rewards[fname][a] = np.load(f"./{fname}/{a}/rewards.npy").tolist()
      except: 
        end_rewards[fname][a] = []
        print(f"Could not find rewards at: ./{fname}/{a}/rewards.npy")

  if 'sac_boid' in brain_names:
    fname = 'sac_boid'
    brains[fname] = {}
    end_rewards[fname] = {}
    for a in agents:
      brains[fname][a] = SAC_Brain(a,0,int(env.boid_state_size),2,sar_env.boid_state,device=device, filepath=fname,update_every=5)
      if not os.path.exists(f"./{fname}/{a}/"): # eps .1 update after 25k
        os.makedirs(f"./{fname}/{a}/")
      try:
        end_rewards[fname][a] = np.load(f"./{fname}/{a}/rewards.npy").tolist()
      except: 
        end_rewards[fname][a] = []
        print(f"Could not find rewards at: ./{fname}/{a}/rewards.npy")

  if 'sac_brain' in brain_names:
    fname = 'sac_brain'
    brains[fname] = {}
    end_rewards[fname] = {}
    for a in agents:
      brains[fname][a] = SAC_Brain(a,0,int(env.vec_state_small_size),2,sar_env.vectorize_state_small,device=device, filepath=fname,update_every=5)
      if not os.path.exists(f"./{fname}/{a}/"): # eps .1 update after 25k
        os.makedirs(f"./{fname}/{a}/")
      try:
        end_rewards[fname][a] = np.load(f"./{fname}/{a}/rewards.npy").tolist()
      except: 
        end_rewards[fname][a] = []
        print(f"Could not find rewards at: ./{fname}/{a}/rewards.npy")
 
  if 'sac_big_brain' in brain_names:
    fname = 'sac_big_brain'
    brains[fname] = {}
    end_rewards[fname] = {}
    for a in agents:
      brains[fname][a] = SAC_Brain(a,0,int(env.vec_state_size),2,sar_env.vectorize_state,device=device, filepath=fname,update_every=5)
      if not os.path.exists(f"./{fname}/{a}/"): # eps .1 update after 25k
        os.makedirs(f"./{fname}/{a}/")
      try:
        end_rewards[fname][a] = np.load(f"./{fname}/{a}/rewards.npy").tolist()
      except: 
        end_rewards[fname][a] = []
        print(f"Could not find rewards at: ./{fname}/{a}/rewards.npy")
  
  if 'sac_radio' in brain_names:
    fname = 'sac_radio'
    brains[fname] = {}
    end_rewards[fname] = {}
    for a in agents:
      brains[fname][a] = SAC_radio(a,0,int(env.vec_state_size),6,sar_env.vectorize_state,device=device, filepath=fname,update_every=5)
      if not os.path.exists(f"./{fname}/{a}/"): # eps .1 update after 25k
        os.makedirs(f"./{fname}/{a}/")
      try:
        end_rewards[fname][a] = np.load(f"./{fname}/{a}/rewards.npy").tolist()
      except: 
        end_rewards[fname][a] = []
        print(f"Could not find rewards at: ./{fname}/{a}/rewards.npy")
  
  
  return brains, end_rewards

if __name__ == "__main__":
  player_num = 0
  agents = ["Human","RoboDog","Drone"]
  max_agents = len(agents)
  pois = ["Child","Child","Adult","Adult","Adult"]
  premade_map = np.load("../LevelGen/Island/Map.npy")
  env = sar_env(max_agents=3,display=True, tile_map=premade_map, agent_names=agents, poi_names=pois,seed=random.randint(0,10000),player=player_num,explore_multiplier=0.1)
  state, info = env.start()
  st = sar_env.boid_state(state,0,True)
  print(f"Message shape: {state['radio']['message'].shape}")
  print(f"State shape: {st.shape}")

  eps = 0
  terminated = False
  # instantiate the policy
  brain_names = ['sac_radio','sac_radio','boid','boid','sac_big_brain']#'torch_sup', 'ppo_big_brain','ppo_brain','ppo_boid',
  brains, end_rewards = load_models(state,agents,env,brain_names)
  # create an optimizer
  # initialize gamma and stats
  n_episode = -1
  render_rate = 100# render every render_rate episodes
  
  while True:
    print(f"Player {env.player}")
    n_episode+=1
    #player_num+=1
    #player_num = player_num%len(agents)
    env.player = player_num
    
    selected_brains = []
    brain = brain_names[random.randint(0,len(brain_names)-1)]
    j = random.randint(0,max_agents-1)
    for i in range(max_agents):
      selected_brains.append(brain_names[random.randint(0,len(brain_names)-1)])
      #if i==j:
        #selected_brains[-1] = 'boid'
    print(f"selected brains: {selected_brains}")
    if n_episode%render_rate==0:
      env.display=True
    else:
      env.display=False
    tot_r = np.zeros(3)
    # reset environment
    state, info = env.start()
    env.debug_render = True
    print('env: ',end='')
    for i in range(len(agents)):
      env.agents[i].brain_name = selected_brains[i]

    while True:
      # render episode every render_rate epsiodes
      # calculate probabilities of taking each action
      np_actions = []
      for i,a in enumerate(agents):
        brains[selected_brains[i]][a].dead = info.agents[i].destroyed
        
        # If an agent is dead just give it 0's for it's actions
        if brains[selected_brains[i]][a].dead:
          np_actions.append(np.zeros((1,14+max_agents)))
          if selected_brains[i] in ['ppo_brain','ppo_boid','ppo_big_brain']:
            brains[selected_brains[i]][a].action(state,i)
          
        else:
          np_actions.append(brains[selected_brains[i]][a].action(state,i))
      np_actions = np.array(np_actions)

      # the brains I made assume states are given in a batch so they return
      # a batch of actions. Because this is a "batch" of size 1, we remove a
      # dimension
      np_actions = np_actions.reshape((np_actions.shape[0],np_actions.shape[2]))
      
      new_state, reward, done, _, info = env.step(np_actions)
      tot_r += reward

      for i,a in enumerate(agents):
        brains[selected_brains[i]][a].update(i,state,np_actions[i],reward[i],new_state,done,_,env)
      
      state = new_state
      if done:
        break

    # Save reward history
    for i,a in enumerate(agents):
      end_rewards[selected_brains[i]][a].append(tot_r[i])
      print(f"{a}({selected_brains[i]}) r: {tot_r[i]}")
    for a in agents:
      for b in brain_names:
        brains[b][a].checkpoint()
        np.save(f"./{b}/{a}/rewards.npy",np.array(end_rewards[b][a]))