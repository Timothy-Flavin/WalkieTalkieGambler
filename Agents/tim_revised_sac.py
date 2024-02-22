import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.distributions import Normal
import copy
import traceback

import numpy as np
import matplotlib.pyplot as plt
import itertools 

from brain import brain

class Actor(nn.Module):
  def __init__(self, state_dim, action_dim, max_action, max_agents=3, device='cpu'):
    super(Actor, self).__init__()
    self.encoder1 = nn.Linear(state_dim, 256)
    self.encoder2 = nn.Linear(256, 256)
    self.action_mu = nn.Linear(256, action_dim)
    self.action_sigma = nn.Linear(256, action_dim)
    self.command_mu = nn.Linear(256+max_agents+8, action_dim+1)
    self.command_sigma = nn.Linear(256+max_agents+8, action_dim+1)
    
    self.talk_layer = nn.Linear(256,2)
    
    self.msg_layer = nn.Linear(256+max_agents, 8)
    self.targ_layer = nn.Linear(256, max_agents)
    
    self.max_action = max_action
    self.mag = 5
    self.to(device)
    self.device=device
    self.max_agents=max_agents 
    # this lets us grab message legality part of the state for masking when taking the state from mem buffer
    self.message_legality_state_offset = state_dim - 8 - max_agents*5 

    #state[self.message_legality_state_offset, self.message_legality_state_offset+8]

  def forward(self, state, legality):
    h = F.relu(self.encoder1(state))
    h = F.relu(self.encoder2(h))
    #print(f"state shape: {state.shape}")
    #print(f"h shape: {h.shape}")
    #print(f"talk layer(h) shape: {self.talk_layer(h).shape}")
    tk = self.talk_layer(h)#5*F.sigmoid(self.talk_layer(h))
    #print(tk)
    talk_dist = torch.softmax(tk, 1) # this is binomial we don't talk about it
    #print("talk_dist")
    #print(talk_dist)
    talk_sampler = Categorical(talk_dist)
    talk = talk_sampler.sample()
    talk_log=talk_sampler.log_prob(talk)
    
    l = legality
    targ = torch.zeros([state.shape[0]]) # integer because it would be sampled
    msg = torch.zeros([state.shape[0]]) #integer because it would be sampled
    command_means = torch.zeros([state.shape[0],3])
    command_stdv = torch.ones([state.shape[0],3])

    targ_dist = torch.ones([state.shape[0],self.max_agents],device=self.device)
    targ_dist[talk>0.5] = torch.softmax(self.targ_layer(h)[talk>0.5], 0) # like this line TODO
    targ_sampler = Categorical(targ_dist)
    targ = targ_sampler.sample()
    #print(f"Talk: {talk}, stayed quiet: {talk<0.5}, num: {(talk<0.5).sum()}")
    targ_log = targ_sampler.log_prob(targ)
    #targ_log[talk<0.5]*=0 #set log prob to zero so prob to 1

    h1 = torch.cat([h,F.one_hot(targ,num_classes=self.max_agents)],1)

    msg_dist = torch.ones([state.shape[0],8]).to(self.device)
    msg_dist[talk>0.5] = torch.softmax((self.msg_layer(h1)*((1-l)*-1000000))[talk>0.5], 0) #mask softmax look again later TODO
    msg_sampler = Categorical(msg_dist)
    msg = msg_sampler.sample()
    msg_log = msg_sampler.log_prob(msg)
    #msg_log[talk<0.5]*=0


    #print(f"""
    #  h: {h.shape},
    #  targ: {F.one_hot(targ,num_classes=self.max_agents).shape}
    #  msg: {F.one_hot(msg,num_classes=8).shape}
    #      """)
    h2 = torch.cat([h,F.one_hot(targ,num_classes=self.max_agents),F.one_hot(msg,num_classes=8)],1)
    #print(f"h2.shape: {h2.shape}")
    command_means = self.command_mu(h2)
    command_stdv = self.command_sigma(h2)

    action_means = self.action_mu(h)
    action_stdv = self.action_sigma(h)
  
    return action_means, action_stdv, command_means, command_stdv, talk, talk_log, msg, msg_log, targ, targ_log


class Critic(nn.Module):
  def __init__(self, state_dim, action_dim, device, max_agents=3):
    super(Critic, self).__init__()
    # Q1 architecture state + (move + command+mag) + talk + msg + target
    #print(action_dim+1 + 1 + 8 + max_agents)
    self.l1 = nn.Linear(state_dim + action_dim*2+1 + 1 + 8 + max_agents, 256)
    self.l2 = nn.Linear(256, 256)
    self.l3 = nn.Linear(256, 1)
    # Q2 architecture
    self.l4 = nn.Linear(state_dim + action_dim*2+1 + 1 + 8 + max_agents, 256)
    self.l5 = nn.Linear(256, 256)
    self.l6 = nn.Linear(256, 1)
    self.to(device)
    self.device=device
  
  def forward(self, state, action, grad=True):
    if grad:
      sa = torch.cat([state, action], 1)

      q1 = F.relu(self.l1(sa))
      q1 = F.relu(self.l2(q1))
      q1 = self.l3(q1)

      q2 = F.relu(self.l4(sa))
      q2 = F.relu(self.l5(q2))
      q2 = self.l6(q2)

      return q1, q2
    else:
      with torch.no_grad():
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

  def Q1(self, state, action):
    sa = torch.cat([state, action], 1)

    q1 = F.relu(self.l1(sa))
    q1 = F.relu(self.l2(q1))
    q1 = self.l3(q1)
    return q1

class SAC_radio(brain):
  def __init__(self, name, anum, state_dim, action_dim, vec, device='cpu', trade_off=0.1, max_action=1, update_every=1,gamma=0.99, tau=0.995, filepath="sac", mag=True,multinomial_noise=0.1):
    super(SAC_radio,self).__init__(name,anum)
    torch.autograd.set_detect_anomaly(True)
    #print(state_dim)
    self.file_name = filepath
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.vec = vec
    self.trade_off = trade_off
    self.max_action = max_action
    self.up_num = 0
    self.gamma = gamma
    self.update_every = update_every
    self.LOG_STD_MAX = 2
    self.LOG_STD_MIN = -20
    self.tau=tau
    self.mag=mag
    self.device = device
    self.multinomial_noise = multinomial_noise

    self.loss_history = []

    self.actor = Actor(state_dim,action_dim,max_action,device=device)
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
    
    self.critic_env = Critic(state_dim,action_dim,device=self.device)
    self.critic_env_optimizer = torch.optim.Adam(self.critic_env.parameters(), lr=3e-4)
    self.critic_env_target = copy.deepcopy(self.critic_env)

    self.critic_msg = Critic(state_dim,action_dim,device=self.device)
    self.critic_msg_optimizer = torch.optim.Adam(self.critic_msg.parameters(), lr=3e-4)
    self.critic_msg_target = copy.deepcopy(self.critic_msg)
    
    self.buffer = ReplayBuffer(state_dim,action_dim, 100000)
    self.norm = Normal(torch.zeros(self.action_dim),torch.ones(self.action_dim))

    self.type_noise = Normal(torch.zeros(8),torch.ones(8)*0.1)
    self.targ_noise = Normal(torch.zeros(3),torch.ones(3)*0.1)

    self.q_params = itertools.chain(self.critic_env.parameters(), self.critic_msg.parameters())

    for name,param in self.actor.named_parameters():
          print(name)
          print(param.data)
    
  def __action__(self,state, legality=None):
    #prob_means,prob_log_stds, msgp,targp = self.actor(state,legality)
    action_means, action_stdv, command_means, command_stdv, talk, talk_log, msg, msg_log, targ, targ_log = self.actor(state,legality)
    
    action_log_stds = torch.clamp(action_stdv,self.LOG_STD_MIN,self.LOG_STD_MAX)
    action_stds = torch.exp(action_log_stds)
    action_dist = Normal(action_means, action_stds)

    command_log_stds = torch.clamp(command_stdv,self.LOG_STD_MIN,self.LOG_STD_MAX)
    command_stds = torch.exp(command_log_stds)
    command_dist = Normal(command_means, command_stds)
    
    action = action_dist.rsample()#self.max_action* #TODO make this not be so ugly
    log_action = action_dist.log_prob(action).sum(axis=-1)
    log_action -= (2*(np.log(2) - action - F.softplus(-2*action))).sum(axis=1)
    command = command_dist.rsample()
    log_command = command_dist.log_prob(command).sum(axis=-1)
    log_command -= (2*(np.log(2) - command - F.softplus(-2*command))).sum(axis=1)
    #command[:,2]+=1 
    #command[:,2]*=self.max_action/2
    action = torch.tanh(action)
    command = torch.tanh(command)
    
    return action, log_action, command, log_command, talk, talk_log, msg, msg_log, targ, targ_log 

  def action(self,state,anum,grad=False, cpu=True, vectorized=False):
    action,command,talk,msg,targ = None,None,None,None,None
    with torch.no_grad():
      vec_state = torch.FloatTensor(self.vec(state,self.anum,True))[None,:].to(self.device)
      legality = torch.FloatTensor(state['legal_messages'][anum])[None,:].to(self.device)
    #else:#print(torch.FloatTensor(vec_state)[None,:].to(self.device))
      action_means, action_stdv, command_means, command_stdv, talk, talk_dist, msg, msg_dist, targ, targ_dist = self.actor(vec_state,legality)
      
      action_log_stds = torch.clamp(action_stdv,self.LOG_STD_MIN,self.LOG_STD_MAX)
      action_stds = torch.exp(action_log_stds)
      action_dist = Normal(action_means, action_stds)

      command_log_stds = torch.clamp(command_stdv,self.LOG_STD_MIN,self.LOG_STD_MAX)
      command_stds = torch.exp(command_log_stds)
      command_dist = Normal(command_means, command_stds)
      
      action = torch.tanh(action_dist.sample())#self.max_action* #TODO make this not be so ugly
      command = torch.tanh(command_dist.sample())
      command[:,2] = (command[:,2]+1)*self.max_action/2

      final_action = torch.cat((action.cpu(),command.cpu(),talk.cpu()[None,:],F.one_hot(msg.cpu(),num_classes=8),F.one_hot(targ.cpu(),num_classes=self.actor.max_agents)),1).numpy()
      #print(f"final action shape {final_action.shape}")
      return final_action
  
  def q_loss(self,states,actions,states_,rewards,dones,legality,anum,critic):
    Q1 = critic.Q1(states,actions)
    
    with torch.no_grad():
      #Get the distribution
      #actions_, act_log_prob, mtp, targp 
      actions_,act_log_prob,command,command_log,talk,talk_log,msg,msg_log,targ,targ_log = self.__action__(states_,legality)

      msg = F.one_hot(msg,8).float()
      targ = F.one_hot(targ,self.actor.max_agents).float()
      msg[talk<0]*=0. # so the Q network get's zeros instead of all 1s. it was 1s earlier to not change log prob
      targ[talk<0]*=0.
      self.type_noise = torch.normal(mean=0, std=self.multinomial_noise, size=msg.shape).to(self.device)
      msg += self.type_noise
      self.targ_noise = torch.normal(mean=0, std=self.multinomial_noise, size=targ.shape).to(self.device)
      targ += self.targ_noise
      talk = talk.float()[:,None]
      noisy_action = torch.cat((actions_,command,talk,msg,targ),1)
      target_Q1, target_Q2 = self.critic_env(states_, noisy_action)
      target_Q = torch.min(target_Q1, target_Q2)

      target = rewards+self.gamma*(1-dones)*(target_Q-self.trade_off*(act_log_prob+command_log))#(target_Q_env-self.trade_off*(act_log_prob+command_log+talk_log,msg_log,targ_log))
      #target_msg = rewards+self.gamma*(1-dones)*target_Q_msg#-self.trade_off*(act_log_prob+command_log+talk_log,msg_log,targ_log))
    critic_loss = ((Q1 - target)**2).mean()

    if abs(critic_loss.item())>1000:
      print(f"log prob sum: ({act_log_prob[0:10]}+{command_log[0:10]}+{talk_log[0:10]}+{msg_log[0:10]}+{targ_log[0:10]})") 
      print(f"rewards: {rewards}") 
      print(f"target q: {target_Q}") 
      input("say something")
    #critic_loss2 = ((Q3 - target_msg)**2).mean()
    
    return critic_loss

  def a_loss(self, states, actions, states_, rewards, dones,legality, anum):
    action,action_log,command,command_log,talk,talk_log,msg,msg_log,targ,targ_log = self.__action__(states, legality)
    #actions = torch.cat([actions], 1)
    tk = (talk>0).int().to(self.device)
    msg = msg*tk # so the Q network get's zeros instead of all 1s. it was 1s earlier to not change log prob
    targ=targ*tk

    act = torch.cat((action,command,targ.float()[:,None],F.one_hot(msg,8),F.one_hot(targ,self.actor.max_agents)),1)
    #for p in self.q_params:
    #  p.requires_grad = False
    Q1, Q2 = self.critic_env(states,act, grad=False)

    Q3, Q4 = self.critic_msg(states,act, grad=False)
    #print(Q1)
    #print(Q2)
    Q_env = torch.minimum(Q1,Q2)
    Q_msg = torch.minimum(Q3,Q4)

    #print(Q_env)
    #print(Q_env.min().item())
    Q_env -= Q_env.min().item()
    Q_env /= max(Q_env.max().item(),0.0001)#torch.maximum(Q_env.max(1,keepdim=True)[0],torch.ones(Q_env.shape).to(self.device)*0.0001)#(Q_env-torch.mean(Q_env,0))/torch.minimum(torch.std(Q_env,0),torch.ones(Q_env.shape).to(self.device)/1000) #TODO make something better
    Q_msg -= Q_msg.min().item()
    Q_msg /= max(Q_msg.max().item(),0.0001)#torch.maximum(Q_msg.max(1,keepdim=True)[0],torch.ones(Q_msg.shape).to(self.device)*0.0001)#(Q_msg-torch.mean(Q_msg,0))/torch.minimum(torch.std(Q_msg,0),torch.ones(Q_msg.shape).to(self.device)/1000) #TODO do better

    action_loss = (self.trade_off*(action_log) - Q_env).mean()
    
    states_where_we_talked = actions[:,3]>0.5
    act_cats = torch.argmax(actions[:,6:6+8],dim=1)
    command_states = torch.logical_and(states_where_we_talked, act_cats == 7)
    going_states = torch.logical_and(states_where_we_talked, act_cats == 6)

    command_loss = (self.trade_off*(command_log[command_states]) -Q_msg).mean() 
    going_loss = (self.trade_off*(command_log[going_states]) -Q_env).mean() 

    talk_loss = (-Q_env*talk_log).mean() #Q_env ro Q_msg test out both TODO
    msg_loss_c = (-Q_msg[command_states]*msg_log[command_states] ).mean() #Q_env ro Q_msg test out both TODO + self.trade_off*msg_log[command_states]
    targ_loss_c = (-Q_msg[command_states]*targ_log[command_states] ).mean() #Q_env ro Q_msg test out both TODO + self.trade_off*msg_log[command_states]

    msg_not_command = torch.logical_and(talk,torch.logical_not(command_states))
    msg_loss = (-Q_env[msg_not_command]*msg_log[msg_not_command] ).mean() #Q_env ro Q_msg test out both TODO + self.trade_off*msg_log[msg_not_command]
    targ_loss = (-Q_env[msg_not_command]*targ_log[msg_not_command] ).mean() #Q_env ro Q_msg test out both TODO + self.trade_off*msg_log[msg_not_command]

    #print(f"Action loss: {action_loss.cpu().item()}")
    #print(f"command loss: {command_loss.cpu().item()}")
    #print(f"going loss: {going_loss.cpu().item()}")
    #print(f"talk loss: {talk_loss.cpu().item()}")
    #print(f"message loss: {msg_loss.cpu().item()}")
    #print(f"message loss c: {msg_loss_c.cpu().item()}")
    #print(f"target loss: {targ_loss.cpu().item()}")
    # print(f"target loss c: {targ_loss_c.cpu().item()}")

    #print(Q_env[0:10])
    #print(Q_msg[0:10])
    # input()
    #print(f"Command states: {command_states}")
    #print(f"Message not commanded: {msg_not_command}")
    #for p in self.q_params:
    #  p.requires_grad = True
    return action_loss, command_loss, going_loss, talk_loss, msg_loss, msg_loss_c, targ_loss, targ_loss_c


  # This is where a model will be trained if in training mode.
  def update(self,anum,state,action,rewards,state_,terminated,truncated,game_instance):
    self.up_num += 1
    self.vec(state,anum,True)
    legality = state['legal_messages'][anum]
    self.buffer.add(self.vec(state,anum,True),action,self.vec(state_,anum,True),rewards,int((terminated or truncated)),legality)
    
    if self.up_num%self.update_every==0:
      #legality = torch.FloatTensor(state['legal_messages'][anum])[None,:].to(self.device)
      states,actions,states_,rewards,dones, legalitys = self.buffer.sample(256)
      #self.up_num = 0
      
      critic_loss1 = self.q_loss(states,actions,states_,rewards,dones,legalitys,anum,self.critic_env)
      self.critic_env_optimizer.zero_grad()
      critic_loss1.backward()
      self.critic_env_optimizer.step()

      #print(states_[:,-self.actor.max_agents])
      critic_loss2 = self.q_loss(states,actions,states_,torch.sum(states_[:,-self.actor.max_agents:],1),dones,legalitys,anum, self.critic_msg)
      self.critic_msg_optimizer.zero_grad()
      critic_loss2.backward()
      self.critic_msg_optimizer.step()

      #update the critic
      (action_loss, 
       command_loss, 
       going_loss, 
       talk_loss, 
       msg_loss, 
       msg_loss_c, 
       targ_loss, 
       targ_loss_c,
       ) = self.a_loss(states,actions,states_,rewards,dones,legalitys,anum)
      
      #self.loss_history.append([
      #  critic_loss1.cpu().item(),
      #  critic_loss2.cpu().item(),
      #  action_loss.cpu().item(), 
      #  command_loss.cpu().item(), 
      #  going_loss.cpu().item(), 
      #  talk_loss.cpu().item(), 
      #  msg_loss.cpu().item(), 
      #  msg_loss_c.cpu().item(), 
      #  targ_loss.cpu().item(), 
      #  targ_loss_c.cpu().item(),
      #])
      #print(f"Loss history")
      strings = ["critic_loss1",
        "critic_loss2",
        "action_loss", 
        "command_loss", 
        "going_loss", 
        "talk_loss", 
        "msg_loss", 
        "msg_loss_c", 
        "targ_loss", 
        "targ_loss_c",]
      #lh = np.array(self.loss_history)
      #if int(self.up_num/self.update_every)%1000 == 0:
      #  for l in range(lh.shape[1]):
      #    plt.plot(lh[:,l], label=strings[l])
      #  plt.legend()
      #  plt.show()

      try:
        self.actor_optimizer.zero_grad()
        action_loss.backward(retain_graph=True) #TODO matthew aggregate these (no questions allowed)
        command_loss.backward(retain_graph=True)
        going_loss.backward(retain_graph=True) 
        talk_loss.backward(retain_graph=True) 
        msg_loss_c.backward(retain_graph=True) 
        msg_loss.backward(retain_graph=True) 
        targ_loss.backward(retain_graph=True) 
        targ_loss_c.backward()
        self.actor_optimizer.step()
      except Exception as e:
        print(e)
        print(f"States {states[0:5].sum()} {states[0:5]},\n actions: {actions[0:5]},\n next_states: {states_[0:5]},\n rewards: {rewards[0:5]},\n Dones: {dones[0:5]},\n legality: {legalitys[0:5]},anum: {anum}")
        for name,param in self.actor.named_parameters():
          print(name)
          print(param.data)
        #input()
        #print(traceback.print_exception(e))

      # spinning up
      with torch.no_grad():
        for p, p_targ in zip(self.critic_env.parameters(), self.critic_env_target.parameters()):
          # NB: We use an in-place operations "mul_", "add_" to update target
          # params, as opposed to "mul" and "add", which would make new tensors.
          p_targ.data.mul_(self.tau)
          p_targ.data.add_((1 - self.tau) * p.data)
        
        for p, p_targ in zip(self.critic_msg.parameters(), self.critic_msg_target.parameters()):
          # NB: We use an in-place operations "mul_", "add_" to update target
          # params, as opposed to "mul" and "add", which would make new tensors.
          p_targ.data.mul_(self.tau)
          p_targ.data.add_((1 - self.tau) * p.data)

  # for algorithms that can only update after an episode
  def update_end_of_episode(self):
    pass
  
  def save(self, filename):
    torch.save(self.critic_env.state_dict(), filename + "_critic_env")
    torch.save(self.critic_env_optimizer.state_dict(), filename + "_critic_env_optimizer")
    torch.save(self.critic_msg.state_dict(), filename + "_critic_msg")
    torch.save(self.critic_msg_optimizer.state_dict(), filename + "_critic_msg_optimizer")
    #print("jappy pini")
    torch.save(self.actor.state_dict(), filename + "_actor")
    torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


  def _load(self, filename):
    self.critic_env.load_state_dict(torch.load(filename + "_critic_env"))
    self.critic_env_optimizer.load_state_dict(torch.load(filename + "_critic_env_optimizer"))
    self.critic_env_target = copy.deepcopy(self.critic_env)

    self.critic_msg.load_state_dict(torch.load(filename + "_critic_msg"))
    self.critic_msg_optimizer.load_state_dict(torch.load(filename + "_critic_msg_optimizer"))
    self.critic_msg_target = copy.deepcopy(self.critic_msg)

    self.actor.load_state_dict(torch.load(filename + "_actor"))
    self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
    #self.actor_target = copy.deepcopy(self.actor)

  # load this model from a checkpoint
  def load(self):
    self._load(f'./{self.file_name}/{self.name}')
  # This will be called every so many minutes to save the model in case
  # of crash or other problems
  def checkpoint(self):   
    self.save(f'./{self.file_name}/{self.name}')

class ReplayBuffer(object):
  def __init__(self, state_dim, action_dim, max_size=int(1e6), radio=True):
    self.max_size = max_size
    self.ptr = 0
    self.size = 0

    self.state = np.zeros((max_size, state_dim))
    self.legality = np.zeros((max_size, 8)) #remove later? TODO
    self.action = np.zeros((max_size, action_dim*3+8+3))
    self.next_state = np.zeros((max_size, state_dim))
    self.reward = np.zeros((max_size, 1))
    self.done = np.zeros((max_size, 1))


    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


  def add(self, state, action, next_state, reward, done, legality):
    self.state[self.ptr] = state
    self.action[self.ptr] = action
    self.next_state[self.ptr] = next_state
    self.reward[self.ptr] = reward
    self.done[self.ptr] = done
    self.legality[self.ptr] = legality

    self.ptr = (self.ptr + 1) % self.max_size
    self.size = min(self.size + 1, self.max_size)


  def sample(self, batch_size):
    ind = np.random.randint(0, self.size, size=batch_size)
    return (
      torch.FloatTensor(self.state[ind]).to(self.device),
      torch.FloatTensor(self.action[ind]).to(self.device),
      torch.FloatTensor(self.next_state[ind]).to(self.device),
      torch.FloatTensor(self.reward[ind]).to(self.device),
      torch.FloatTensor(self.done[ind]).to(self.device),
      torch.FloatTensor(self.legality[ind]).to(self.device)
    )