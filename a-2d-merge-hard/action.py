from highway_env.envs.common.action import ActionType
from gym.spaces import Discrete
from settings import motor_model
from highway_env.vehicle.kinematics import Vehicle
from settings import numHA, pv_stddev
import numpy as np
from observation_helper import process_obs

class MyAction(ActionType):


    def space(self):
        return Discrete(numHA)

    @property
    def vehicle_class(self):
        return Vehicle

    def act(self, action: int) -> None:
        obs = process_obs(self.env, self.env.full_observation_type.observe())
        prev_la = self.env.repo["la"]
        la = motor_model(action, obs, prev_la)

        la[0] += np.random.normal(0, pv_stddev[0])
        la[1] += np.random.normal(0, pv_stddev[1])

        self.controlled_vehicle.act({
                "steering": la[0],
                "acceleration": la[1]
            })
        self.env.repo["la"] = la
        self.env.repo["ha"] = action




def action_factory(env: 'AbstractEnv', config: dict) -> ActionType:
  assert(config["type"]=="myact")
  return MyAction(env, **config)