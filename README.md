# WalkieTalkieGambler
RL project on a custom environment to do communication

# For Ethan and Matthew

## Installing Halsarc
https://test.pypi.org/project/halsarc/

Pip install this package and then create a project with the folders `\Agents` and `\LevelGen` in it based on the specifications in Halsarc to test, or clone this repo and create your own branch to make changes and pull requests.

## Already trained models

Everything you might be interested in is inside the Agents folder. I made every agent follow a "Brain" api defined in brain.py so that we can have many kinds of models play together. 

train_models.py gives an example of creating many different models with different brains. SAC_Radio.py is the soft actor critic with some radio capabilities that you can build from, or start your own. The giant long function full of constructors is just a lazy version of a factory so you can probably ignore it. 

`encoding.txt` is a document specifying the state returned by the environment and what everything means with actions and the state. The state is not a numpy array by default because it holds a lot information that you may want to use or ignore depending on your model.

## Easier environment option

https://github.com/Timothy-Flavin/MATCH

The marl gridworld2 env in this repo is simpler than searh and rescue and will run faster. Similar to the icy lake env in Gym, it has 4 agents and each can send a command at a given timestep. If you want to start with this environment while HalSarc is being updated that is very reasonable. If so I recommend grabbing the SAC_radio code and marlgridworld2.py and editing the gridworld and sac code at the same time to play around with them. 

## Debugging this or halsarc

If halsarc is doing unexpected behavior, feel free to clone the repo yourself and change the import statements to import from your local directory instead of from `Halsarc import blah`, or alternatively select halsarc objects and go to definition and change your local install. If you do the first thing then feel free to make pull requests or reach out to me, if you do the second just let me know what was broken or what seemed broken. 

Ask questions, if you have them, then any other user would too so I'll try to add the answers to some kind of docs.