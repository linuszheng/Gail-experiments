import numpy as np
import gym
from gym import spaces

EPSILON = 10E-10

class Env_1d(gym.Env):

    def __init__(self):
      self.dt = .1
      self.pos = 0.
      self.vel = 0.
      self.acc = 0.
      self.prev_acc = 0.
      self.prev_ha = 0.
      self.decMax = -5.
      self.accMax = 6.
      self.vMax = 10.
      self.target = 100.
      self.observation_space = spaces.Box( low=np.array([0, -60, 0, 0, 0, 100]+[-3]+[0,0,0]), 
                                  high=np.array([200, 0, 60, 120, 120, 200]+[3]+[1,1,1]), 
                                  shape=(10,), dtype=np.float32)
      self.action_space = spaces.Discrete(3)
      self.t = 0

    def config(self, decMax, accMax, vMax, target):
      self.decMax = float(decMax)
      self.accMax = float(accMax)
      self.vMax = float(vMax)
      self.target = float(target)

    def _get_info(self):
      return {}
    
    def motor_model_possibilities(self):
      acc0 = min(self.acc+1, self.accMax)
      acc2 = max(self.acc-1, self.decMax)
      acc1 = acc0 if self.acc<=0 else acc2
      return [acc0, acc1, acc2]

    def _get_obs(self):
      one_hot = self.prev_ha
      return np.array([self.pos, self.decMax, self.accMax, self.vMax, self.vel, self.target] 
      + self.acc + one_hot, dtype=np.float64)


    def reset(self, seed=None, options=None):
      # super().reset(seed=seed)
      self.pos = 0.
      self.vel = 0.
      self.acc = 0.
      self.t = 0.
      self.prev_ha = 0.
      return self._get_obs()

    def step(self, action):
      self.prev_ha = action[0]
      prev_vel = self.vel
      # self.acc = action[0] + np.random.normal(0, 1)
      self.acc = self.motor_model_possibilities[action[0]]
      self.vel = self.vel+self.acc*self.dt
      if self.vel < EPSILON:
        self.vel = 0
      if abs(self.vel - self.vMax) < EPSILON:
        self.vel = self.vMax
      if abs(self.pos - self.target) < EPSILON:
        self.pos = self.target
      self.pos += (prev_vel + self.vel)*.5*self.dt
      self.t += 1
      return self._get_obs(), 0, self.t > MAX_T, self._get_info()




