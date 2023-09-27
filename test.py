import numpy as np
import halsarc

#from Game import sar_env
#from Agent import player_controller

agents = ["Human","RoboDog","Drone"]
pois = ["Child", "Child", "Adult"]
premade_map = np.load("../LevelGen/Island/Map.npy")
game = sar_env(display=True, tile_map=premade_map, agent_names=agents, poi_names=pois,player=0)
state, info = game.start()
controller = player_controller(None)
terminated = False

while not terminated:
  actions = np.zeros((len(agents),14+len(agents)))
  for i,a in enumerate(agents):
    actions[i,0:2] = controller.choose_action(state=state, game_instance=game)
    actions[i,2:] = np.zeros(12+len(agents))
  state, rewards, terminated, truncated, info = game.step(actions=actions)
  game.wait(100)