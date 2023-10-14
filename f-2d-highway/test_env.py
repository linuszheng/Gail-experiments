
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
from gail import GAIL
from nets import MyRewardNet
from settings import numHA, _n_timesteps, motor_model, pv_stddev, initialHA, initialLA, _min_performance_to_save
from scipy.stats import norm
from numpy import random
import torch
from hyperparams import _max_disc_acc_until_quit, _max_mode_until_quit, _learning_rate_func, \
_n_gen_train_steps, _n_disc_updates_per_round, _buf_multiplier, _policy_net_shape, \
_ent_coef_lo, _ent_coef_hi, _ent_coef_slope_start, _ppo_settings


min_performance_to_save =  _min_performance_to_save
model_save_name = "models/model-"+str(random.randint(0,100000000000))


m_n_real_to_fake_label_flip = 0
try:
  from hyperparams import _n_real_to_fake_label_flip
  m_n_real_to_fake_label_flip = _n_real_to_fake_label_flip
except:
  pass



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
  data = get_single_expert_df(n).astype({"HA": int})
  ha = data[_ha_column].to_numpy()
  la = data[_la_column].to_numpy()
  features = data[_feature_column].to_numpy()
  all_obs = data[:][:_n_timesteps].to_numpy()
  # features = np.log1p(features)
  # all_obs = np.log1p(all_obs)

  ha = np.append(np.array([[initialHA]]), ha, axis=0)[:-1]    # gives obs the prev HA instead of current
  la = np.append(np.array([initialLA]), la, axis=0)[:-1]    # gives obs the prev LA instead of current
  obs = np.concatenate((features, la), axis=1)



  dones = np.array([False]*_n_timesteps)
  cur_obs = obs[:_n_timesteps]
  next_obs = obs[1:_n_timesteps+1]


  ha_hidden = np.array([0]*_n_timesteps)          # IGNORED IN REWARD NET
  zeros = np.array([[0]*numHA]*_n_timesteps)     # IGNORED IN REWARD NET

  ha = np.hstack(ha)
  ha_one_hot = np.eye(numHA)[ha][:_n_timesteps]

  obs_with_ha_gt = np.concatenate((cur_obs, ha_one_hot), axis=1)
  obs_with_ha_hidden = np.concatenate((cur_obs, zeros), axis=1)
  next_obs = np.concatenate((next_obs, zeros), axis=1)

  actual_ha = data[_ha_column][:_n_timesteps].to_numpy()




  return {
    "obs": obs_with_ha_hidden,
    "next_obs": next_obs,
    "acts": ha_hidden,
    "dones": dones,
    "actual_ha": actual_ha,
    "all_obs": all_obs,
    "gt_obs": obs_with_ha_gt
  }




def get_err(a, b, stdev):
  return np.log(1-norm.cdf(abs(a-b)/stdev))

def evaluate(model, trajectories):
  sum_actual_err = 0
  sum_pred_err = 0
  sum_wrong = 0
  sum_right = 0
  sum_pred_err3 = 0
  sum_wrong3 = 0
  sum_right3 = 0
  ha_chosen = [0] * numHA
  for i, traj in enumerate(trajectories):
    print(f"DATA ENV {i}")
    last_ha = initialHA
    for all_features, select_features, ha in zip(traj["all_obs"], traj["gt_obs"], traj["actual_ha"]):
      predicted_ha = model.predict(select_features)
      la1 = motor_model(ha[0], all_features, select_features[5:7])
      la2 = motor_model(predicted_ha[0], all_features, select_features[5:7])
      print(f"ha, pred1, pred3:             {ha[0]}", end="  ")
      print(f"{predicted_ha[0]}", end="  ")
      la_actual = all_features[-3:-1]
      actual_err = get_err(la1[0], la_actual[0], pv_stddev[0]) + get_err(la1[1], la_actual[1], pv_stddev[1])
      pred_err = get_err(la2[0], la_actual[0], pv_stddev[0]) + get_err(la2[1], la_actual[1], pv_stddev[1])
      actual_err = max(actual_err, -10)
      pred_err = max(pred_err, -10)
      sum_actual_err += actual_err
      sum_pred_err += pred_err
      sum_wrong += (ha[0]!=predicted_ha[0])
      sum_right += (ha[0]==predicted_ha[0])

      one_hot = np.zeros(4)
      np.put(one_hot,last_ha,1)
      obs_3 = np.concatenate((select_features[:-4], one_hot))
      pred_ha_3 = model.predict(obs_3)
      la3 = motor_model(pred_ha_3[0], all_features, select_features[5:7])
      print(f"{pred_ha_3[0]}")
      pred_err3 = get_err(la3[0], la_actual[0], pv_stddev[0]) + get_err(la3[1], la_actual[1], pv_stddev[1])
      pred_err3 = max(pred_err3, -10)
      sum_pred_err3 += pred_err3
      sum_wrong3 += (ha[0]!=pred_ha_3[0])
      sum_right3 += (ha[0]==pred_ha_3[0])
      last_ha = pred_ha_3[0]

      ha_chosen[predicted_ha[0]] += 1
      ha_chosen[pred_ha_3[0]] += 1

  print()
  print("AVG ACTUAL ERR.               " + str(sum_actual_err / _n_timesteps / 30))
  print("AVG PRED ERR 1.               " + str(sum_pred_err / _n_timesteps / 30))
  print("AVG PRED ERR 3.               " + str(sum_pred_err3 / _n_timesteps / 30))
  print("ACC 1.                        " + str(sum_right/(sum_right+sum_wrong)))
  print("ACC 3.                        " + str(sum_right3/(sum_right3+sum_wrong3)))

  acc = sum_right3/(sum_right3+sum_wrong3)
  global min_performance_to_save, model_save_name
  if acc >= min_performance_to_save:
    _learner.policy.save(model_save_name+"-acc-"+str(acc))
    print("saved model to "+model_save_name)
    min_performance_to_save = acc

  ha_chosen = [float(val) / float(2*_n_timesteps*30) for val in ha_chosen]
  print()
  print("distribution of HA choices")
  print(ha_chosen)
  print()
  return max(ha_chosen)








_traj_train = [get_single_expert_traj(i) for i in range(10)]
_traj_all = [get_single_expert_traj(i) for i in range(30)]

register(id="merge-v0", entry_point='custom_envs.envs:Env_2d_merge')


_env_test = gym.make("merge-v0", config={"simulation_frequency": 24,
  "policy_frequency": 8,
  "lanes_count": 4,
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
  "lanes_count": 4,
  "initial_lane_id": 0,
  'vehicles_count': 50,
  "duration": _n_timesteps})



# GPU ----------------------------------------------------------------------------
import torch

mem_amount = [0]*8
gpus_avail = [0, 1, 2, 3, 4, 5, 6, 7]
for i in range(8):
  mem_amount[i] = torch.cuda.mem_get_info(gpus_avail[i])
  print(f"gpu {gpus_avail[i]} has {mem_amount[i]} mem available")
chosen_gpu = gpus_avail[mem_amount.index(max(mem_amount))]
device = torch.device(f'cuda:{chosen_gpu}')
print('Using device:', device)
print()




# train --------------------------------------------------------------------------------
_n_train_loops = 100000
_learner = PPO(
    env=_venv,
    policy=ActorCriticPolicy,
    batch_size=_n_timesteps,
    ent_coef=_ppo_settings["ent_coef"],
    learning_rate=_ppo_settings["learning_rate"],
    n_epochs=_ppo_settings["n_epochs"],
    gamma=_ppo_settings["gamma"],
    n_steps=_n_timesteps,
    device=device,
    policy_kwargs={
      "net_arch": _policy_net_shape
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
    demo_batch_size=_n_timesteps,                                         # cons
    gen_replay_buffer_capacity=_n_gen_train_steps*_buf_multiplier,      # cons
    gen_train_timesteps=_n_gen_train_steps,                             # gen per round
    n_disc_updates_per_round=_n_disc_updates_per_round,                 # disc per round
    venv=_venv,
    gen_algo=_learner,
    reward_net=_reward_net,
    n_real_to_fake_label_flip=int(m_n_real_to_fake_label_flip*_n_timesteps)
)

evaluate(_learner, _traj_all)
sanity(_learner)
for i in range(_n_train_loops):
    print("LOOP # "+str(i))
    train_info = _gail_trainer.train(_n_gen_train_steps)
    mode_percentage = evaluate(_learner, _traj_all)
    sanity(_learner)
    if train_info["disc_acc"]>=_max_disc_acc_until_quit:
      print(f"FALSE CONVERGENCE ({train_info['disc_acc']:.5f}>={_max_disc_acc_until_quit:.5f}). terminating program.")
      quit()
    if mode_percentage>=_max_mode_until_quit:
      print(f"MODE COLLAPSE ({mode_percentage:.5f}>={_max_mode_until_quit:.5f}). terminating program.")
      quit()
    if mode_percentage>=_ent_coef_slope_start:
      print(f"NEAR MODE COLLAPSE ({mode_percentage:.5f}>={_ent_coef_slope_start:.5f}). RAISE ENTROPY")
      dif_percentage = (mode_percentage-_ent_coef_slope_start)/(1-_ent_coef_slope_start)
      _learner.ent_coef = _ent_coef_lo + dif_percentage*(_ent_coef_hi-_ent_coef_lo)
    else:
      _learner.ent_coef = _ent_coef_lo

