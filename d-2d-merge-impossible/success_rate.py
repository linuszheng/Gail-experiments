from stable_baselines3.ppo import MlpPolicy
import torch
from settings import _n_timesteps, laneFinder, lanes_count
import gym
import my_spaces
from gym.envs.registration import register
import numpy as np

np.set_printoptions(suppress=True, precision=3)


import warnings
warnings.filterwarnings("ignore")

device = torch.device('cpu')
my_model = MlpPolicy.load("models/model-3940627136-acc-0.7067567567567568", device=device)
register(id="merge-v0", entry_point='custom_envs.envs:Env_2d_merge')





_env_test = gym.make("merge-v0", config={"simulation_frequency": 24,
  "policy_frequency": 8,
  "lanes_count": lanes_count,
  "initial_lane_id": 0,
  'vehicles_count': 50,
  "duration": _n_timesteps})
  



def sanity(model):
  prev_obs = _env_test.reset()
  crashed = False
  for i in range(0,_n_timesteps):
    predicted_action = model.predict(prev_obs)[0]
    # print(predicted_action)
    cur_state = _env_test.step(predicted_action)
    prev_obs = cur_state[0]
    if _env_test.vehicle.crashed and not crashed:
      print("CRASHED")
      crashed = True
    if not _env_test.vehicle.on_road and not crashed:
      print("OFF ROAD")
      crashed = True
  print("LANE "+str(_env_test.observation_type.get_lane()))
  if _env_test.observation_type.get_lane() == lanes_count-1:
    print("SUCCESS")
    return True
  else:
    print("FAILURE")
    return False


successes = 0
for i in range(100):
  successes += sanity(my_model)
print("SUCCESS RATE")
print(successes / 100.0)


