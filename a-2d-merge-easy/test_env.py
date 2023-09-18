
import gym
from highway_env.envs import highway_env
import my_spaces
from gym.envs.registration import register
import pandas as pd
import numpy as np
from imitation.util.util import make_vec_env
from imitation.rewards.reward_nets import BasicRewardNet
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from imitation.util.networks import RunningNorm
from imitation.algorithms.adversarial.gail import GAIL
from nets import MyRewardNet
from settings import numHA, _n_timesteps, motor_model, pv_stddev
from scipy.stats import norm


import warnings
warnings.filterwarnings("ignore")


_rng = np.random.default_rng()
_ha_column = ["HA"]
_la_column = ["LA.steer", "LA.acc"]
_feature_column = ["x", "vx", "l_x", "f_x", "r_x"]
_data_path = "data"

np.set_printoptions(suppress=True, precision=3)






# get expert data -----------------------------------------------------------------
def get_single_expert_df(n):
  return pd.read_csv(_data_path+f"/data{n}.csv", skipinitialspace=True)
def get_single_expert_traj(n):
  data = get_single_expert_df(n)
  ha = data[_ha_column].to_numpy()
  la = data[_la_column].to_numpy()
  features = data[_feature_column].to_numpy()

  la = np.append(np.array([[0,0]]), la, axis=0)[:-1]    # gives obs the prev LA instead of current
  obs = np.concatenate((features, la), axis=1)



  dones = np.array([False]*_n_timesteps)
  cur_obs = obs[:_n_timesteps]
  next_obs = obs[1:_n_timesteps+1]


  ha = np.array([0]*_n_timesteps)          # IGNORED IN REWARD NET
  zeros = np.array([[0]*numHA]*_n_timesteps)     # IGNORED IN REWARD NET
  cur_obs = np.concatenate((cur_obs, zeros), axis=1)
  next_obs = np.concatenate((next_obs, zeros), axis=1)

  actual_ha = data[_ha_column][:_n_timesteps].to_numpy()
  all_obs = data[:][:_n_timesteps].to_numpy()


  return {
    "obs": cur_obs,
    "next_obs": next_obs,
    "acts": ha,
    "dones": dones,
    "actual_ha": actual_ha,
    "all_obs": all_obs
  }




def get_err(a, b, stdev):
  return np.log(1-norm.cdf(abs(a-b)/stdev))

def evaluate(model, trajectories):
  sum_actual_err = 0
  sum_pred_err = 0
  sum_wrong = 0
  sum_right = 0
  for i, traj in enumerate(trajectories):
    print(f"DATA ENV {i}")
    for all_features, select_features, ha in zip(traj["all_obs"], traj["obs"], traj["actual_ha"]):
      predicted_ha = model.predict(select_features)
      la1 = motor_model(ha[0], all_features, select_features[5:7])
      la2 = motor_model(predicted_ha[0], all_features, select_features[5:7])
      print(f"actual: {ha[0]} ---- pred: {predicted_ha[0]}")
      la_actual = all_features[-3:-1]
      actual_err = get_err(la1[0], la_actual[0], pv_stddev[0]) + get_err(la1[1], la_actual[1], pv_stddev[1])
      pred_err = get_err(la2[0], la_actual[0], pv_stddev[0]) + get_err(la2[1], la_actual[1], pv_stddev[1])
      actual_err = max(actual_err, -10)
      pred_err = max(pred_err, -10)
      # print()
      sum_actual_err += actual_err
      sum_pred_err += pred_err
      sum_wrong += (ha[0]!=predicted_ha[0])
      sum_right += (ha[0]==predicted_ha[0])
  print("AVG ACTUAL ERR " + str(sum_actual_err / _n_timesteps / 30))
  print("AVG PRED ERR " + str(sum_pred_err / _n_timesteps / 30))
  print("ACC " + str(sum_right/(sum_right+sum_wrong)))
  print()
  print()









_traj_train = [get_single_expert_traj(i) for i in range(10)]
_traj_all = [get_single_expert_traj(i) for i in range(30)]

register(id="merge-v0", entry_point='custom_envs.envs:Env_2d_merge')


_env_test = gym.make("merge-v0", config={"simulation_frequency": 24,
  "policy_frequency": 8,
  "lanes_count": 6,
  "initial_lane_id": 0,
  'vehicles_count': 50,
  "duration": _n_timesteps})

def sanity(model):
  print("SANITY")
  prev_obs = _env_test.reset()
  for i in range(0,_n_timesteps):
    predicted_action = model.predict(prev_obs)[0]
    print(predicted_action)
    cur_state = _env_test.step(predicted_action)
    prev_obs = cur_state[0]
    if _env_test.vehicle.crashed:
      print("CRASHED")
      break
    if not _env_test.vehicle.on_road:
      print("OFF ROAD")
      break
    print([float(f"{num:.3f}") for num in prev_obs])
  print()



_venv = make_vec_env("merge-v0", n_envs=10, rng=_rng)
_venv.env_method("configure", {"simulation_frequency": 24,
  "policy_frequency": 8,
  "lanes_count": 6,
  "initial_lane_id": 0,
  'vehicles_count': 50,
  "duration": _n_timesteps})



_n_train_loops = 100000
_n_train_steps = 40
_learner = PPO(
    env=_venv,
    policy=ActorCriticPolicy,
    batch_size=_n_timesteps,
    ent_coef=0.00005,
    learning_rate=0.0005,
    n_epochs=10,
    gamma=.999,
    n_steps=_n_timesteps,
    policy_kwargs={
      "net_arch": dict(pi=[16, 16, 16], vf=[16, 16, 16])
    }
)
_reward_net = MyRewardNet(
    _venv.observation_space,
    _venv.action_space,
    normalize_input_layer=RunningNorm,
    features_to_use=list(range(5)),
    la_indices=[5,6]
)
_gail_trainer = GAIL(
    demonstrations=_traj_train,
    demo_batch_size=_n_timesteps,                 # cons
    gen_replay_buffer_capacity=40,                # cons
    gen_train_timesteps=40,                       # gen per round
    n_disc_updates_per_round=3,                   # disc per round
    venv=_venv,
    gen_algo=_learner,
    reward_net=_reward_net,
)

evaluate(_learner, _traj_all)
sanity(_learner)
for i in range(_n_train_loops):
    print("LOOP # "+str(i))
    _gail_trainer.train(_n_train_steps)           # total rounds = _n_train_steps / gen_train_timesteps
    evaluate(_learner, _traj_all)
    sanity(_learner)



