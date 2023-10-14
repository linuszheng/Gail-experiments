from stable_baselines3.ppo import MlpPolicy
import torch
from settings import n_timesteps
import gym
from gym.envs.registration import register
import numpy as np

np.set_printoptions(suppress=True, precision=3)


import warnings
warnings.filterwarnings("ignore")

device = torch.device('cpu')
my_model = MlpPolicy.load("models/model-4977802292-acc-0.9333333333333333", device=device)
register(id="env-v1", entry_point='custom_envs.envs:Env_1d')





_env_test = gym.make("env-v1")




def sanity(model):
  prev_obs = _env_test.reset()
  for i in range(0,n_timesteps):
    predicted_action = model.predict(prev_obs)[0]
    cur_state = _env_test.step(predicted_action)
    prev_obs = cur_state[0]
  print(str(_env_test.pos)+" "+str(_env_test.vel))
  if _env_test.pos < 140 and _env_test.pos > 60 and abs(_env_test.vel) < 0.1:
    print("SUCCESS")
    return True
  else:
    print("FAILURE")
    return False


r_states = [
         (6, -5, 10, 100),
        (15, -3, 40, 100),
        (8, -9, 20, 100),
        (9, -6, 12, 100),
        (7, -10, 25, 100),
        (5, -7, 30, 100),
        (10, -15, 30, 100),
        (12, -10, 25, 100),
        (13, -20, 40, 100),
        (5, -8, 15, 100),
        (10, -5, 20, 100),
        (5, -8, 10, 100),
        (4, -4, 30, 100),
        (4, -4, 5, 100),
        (10, -10, 50, 100),
        (30, -30, 200, 100),
        (30, -24, 11, 100),
        (8, -27, 38, 100),
        (27, -16, 11, 100),
        (10, -21, 28, 100),
        (28, -28, 23, 100),
        (16, -9, 81, 100),
        (19, -26, 67, 100),
        (10, -17, 7, 100),
        (19, -25, 27, 100),
        (20, -8, 10, 100),
        (27, -20, 72, 100),
        (21, -7, 99, 100),
        (19, -12, 12, 100),
        (10, -12, 10, 100)
]


successes = 0
for i in range(100):
  r_state = r_states[i % 30]
  _env_test.config(r_state[1], r_state[0], r_state[2], r_state[3])
  successes += sanity(my_model)
print("SUCCESS RATE")
print(successes / 100.0)


