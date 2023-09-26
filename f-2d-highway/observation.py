from highway_env.envs.common.observation import ObservationType
from gym.spaces import Box, Discrete, Dict
import numpy as np
from observation_helper import process_obs
from settings import feature_indices
from settings import numHA



class MyObservation(ObservationType):
  def __init__(self, env, base_obs):
    super().__init__(env)
    self.base_obs = base_obs
  def space(self):
      return Box(                   low=np.array([0, 0, 0, 0, 0]+[-.3, -30]+[0,0,0,0]), 
                                  high=np.array([600, 60, 600, 600, 600]+[.3, 30]+[1,1,1,1]), 
                                  shape=(11,), dtype=np.float32)
  def observe(self):
    res = process_obs(self.env, self.base_obs.observe())
    one_hot = np.zeros(numHA)
    np.put(one_hot,self.env.repo["ha"],1)
    combo_obs = np.concatenate((res[feature_indices], self.env.repo["la"], one_hot))
    return combo_obs
