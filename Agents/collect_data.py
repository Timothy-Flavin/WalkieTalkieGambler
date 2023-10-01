from halsarc.Game.game import sar_env
from halsarc.Game.record_player import player_recorder
import numpy as np

if __name__=="__main__":
  premade_map = np.load('../LevelGen/Island/Map.npy')
  print(premade_map.shape)
  rcdr = player_recorder(player=2,
                         agents=['Human','Drone','RoboDog'],
                         pois=['Child','Child','Adult'],
                         premade_map=premade_map,
                         max_agents=3,
                         data_folder="./recorded_data/",
                         fdirs=["./nets/","./trees/"])
  rcdr.record()