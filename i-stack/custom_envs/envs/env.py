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
    self.observation_space = spaces.Box(   low=np.array([-1]*16 + [-2]*4 + [0]*5), 
                                  high=np.array([1]*16 + [2]*4 + [1]*5), 
                                  shape=(25,), dtype=np.float32)
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
    target_bottom = observation["desired_goal"][0:3]
    target_top = observation["desired_goal"][3:6]

    x, y, z, end_width = world_state[0], world_state[1], world_state[2], world_state[6]
    bx1, by1, bz1, bx2, by2, bz2 = world_state[7], world_state[8], world_state[9], world_state[19], world_state[20], world_state[21]
    tx1, ty1, tz1, tx2, ty2, tz2 = target_bottom[0], target_bottom[1], target_bottom[2], target_top[0], target_top[1], target_top[2]

    bx1, by1, bz1, bx2, by2, bz2 = bx1 - x, by1 - y, bz1 - z, bx2 - x, by2 - y, bz2 - z
    tx2, ty2, tz2 = tx2 - x, ty2 - y, tz2 - z

    obs_pruned = [x, y, z, end_width, bx1, by1, bz1, bx2, by2, bz2, tx1, ty1, tz1, tx2, ty2, tz2]
        
    
    one_hot = [0,0,0,0,0]
    one_hot[self.last_ha] = 1
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
