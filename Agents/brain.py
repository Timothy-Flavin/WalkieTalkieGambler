class brain():
  # takes the state and calculates an action for a given agent. 
  def action(self,state,anum):
    pass
  
  # This is where a model will be trained if in training mode.
  def update(self,anum,state,rewards,terminated,truncated,game_instance):
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