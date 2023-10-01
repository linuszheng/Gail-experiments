import numpy as np
import gym
from gym import spaces
from settings import numHA, n_timesteps, pv_stddev, pv_range, initialHA, initialLA, motor_model
from panda_gym.envs import PandaStackEnv
from gym.utils import seeding

EPSILON = 10E-10

class Env(gym.Env):

  panda_env = PandaStackEnv()

  def __init__(self):
    self.observation_space = spaces.Box(   low=np.array([-1]*10+[-4, -4, -4, -4]+[0,0]), 
                                  high=np.array([1]*10+[4, 4, 4, 4]+[1,1]), 
                                  shape=(16,), dtype=np.float32)
    self.action_space = spaces.Discrete(numHA)
    self.reset()
    
  
  def reset(self):
    self.panda_env.reset()
    self.t = 0
    self.last_ha = initialHA
    self.last_la = initialLA
    return self._get_obs()

  def _get_obs(self):
    observation = self.panda_env._get_obs()
    world_state = observation["observation"]
    target_pos = observation["desired_goal"][0:3]
    x, y, z, bx, by, bz, tx, ty, tz, end_width = world_state[0], world_state[1], world_state[2], world_state[7], world_state[8], world_state[9], target_pos[0], target_pos[1], target_pos[2], world_state[6]
    obs_pruned = [x, y, z, bx, by, bz, tx, ty, tz, end_width]
    one_hot = [0,1] if self.last_ha else [1,0]
    return np.concatenate((obs_pruned, self.last_la, one_hot))

  def _get_info(self):
    return {}

  def step(self, ha_to_take):
    self.t += 1
    la_to_take = motor_model(self.last_ha, self._get_obs(), self._get_obs())
    self.panda_env.step(la_to_take)
    self.last_ha = ha_to_take
    self.last_la = la_to_take
    return self._get_obs(), 0, self.t > n_timesteps, {}
