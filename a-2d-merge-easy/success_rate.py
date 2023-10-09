from stable_baselines3.ppo import MlpPolicy
import torch
from settings import _n_timesteps
import gym
from gym.envs.registration import register
import my_spaces



import warnings
warnings.filterwarnings("ignore")

device = torch.device('cpu')
my_model = MlpPolicy.load("models/model-38261234681-acc-0.7040540540540541", device=device)
register(id="merge-v0", entry_point='custom_envs.envs:Env_2d_merge')





_env_test = gym.make("merge-v0", config={"simulation_frequency": 24,
  "policy_frequency": 8,
  "lanes_count": 6,
  "initial_lane_id": 0,
  'vehicles_count': 50,
  "duration": _n_timesteps})
  



def sanity(model):
  prev_obs = _env_test.reset()

  for i in range(0,_n_timesteps):
    predicted_action = model.predict(prev_obs)[0]
    # print(predicted_action)
    cur_state = _env_test.step(predicted_action)
    prev_obs = cur_state[0]
    if _env_test.vehicle.crashed:
      print("CRASHED")
      return False
    if not _env_test.vehicle.on_road:
      print("OFF ROAD")
      return False
  print("SUCCESS")
  return True


successes = 0
for i in range(100):
  successes += sanity(my_model)
print("SUCCESS RATE")
print(successes / 100.0)


