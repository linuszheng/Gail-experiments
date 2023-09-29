import numpy as np
import gym
from gym import spaces
from settings import numHA, n_timesteps, pv_stddev, pv_range


EPSILON = 10E-10

class Env(PandaPickAndPlaceEnv):

  def __init__(self):
    self.observation_space = spaces.Box(   low=np.array([-1]*9+[-4, -4, -4, -4]+[0,0]), 
                                  high=np.array([1]*9+[4, 4, 4, 4]+[1,1]), 
                                  shape=(15,), dtype=np.float32)
    self.action_space = spaces.Discrete(numHA)
    self.reset()
    
  
  def reset(self):
    self.t = 0
    self.last_ha = 0
    self.last_la = 0
    return super.reset()



  def step(self):
    self.t += 1
  
