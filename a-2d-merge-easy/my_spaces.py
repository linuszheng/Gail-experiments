from observation import MyObservation
from action import MyAction
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import KinematicObservation



def new_define_spaces(self):


        self.full_observation_type = KinematicObservation(
              self, 
              vehicles_count=10,
              features=['presence', 'x', 'y', 'vx', 'vy', 'heading'],
              absolute=True)
        self.full_observation_space = self.full_observation_type.space()

        self.observation_type = MyObservation(
                self, 
                self.full_observation_type)
        self.observation_space = self.observation_type.space()

        self.action_type = MyAction(self)
        self.action_space = self.action_type.space()


KinematicObservation.normalize_obs = lambda self, df: df
AbstractEnv.define_spaces = new_define_spaces