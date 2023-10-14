from stable_baselines3.ppo import MlpPolicy
import torch
from settings import n_timesteps
import gym
from gym.envs.registration import register
# import my_spaces



import warnings
warnings.filterwarnings("ignore")

device = torch.device('cpu')
my_model = MlpPolicy.load("models/model-31222174559-acc-0.4419463087248322", device=device)




register(id="env-v0", entry_point='custom_envs.envs:Env')
_env_test = gym.make("env-v0")
  



def sanity(model):
  print("SANITY")
  prev_obs = _env_test.reset()
  for i in range(0,n_timesteps):
    predicted_action = model.predict(prev_obs)[0]
    cur_state = _env_test.step(predicted_action)
    prev_obs = cur_state[0]
    success = cur_state[1]
    if success:
      print("SUCCESS")
      return True
  print("FAILURE")
  return False


successes = 0
for i in range(100):
  successes += sanity(my_model)
print("SUCCESS RATE")
print(successes / 100.0)


