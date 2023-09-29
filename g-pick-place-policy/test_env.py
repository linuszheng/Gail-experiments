
import gym
from highway_env.envs import highway_env
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
from settings import numHA, n_timesteps, motor_model, pv_stddev, initialHA, initialLA, la_idx_wrt_lim_obs, la_idx_wrt_all_obs, features_idx, training_set
from scipy.stats import norm
import torch
from hyperparams import _max_disc_acc_until_quit, _max_mode_until_quit, _learning_rate_func, \
_n_gen_train_steps, _n_disc_updates_per_round, _buf_multiplier, _policy_net_shape, \
_ent_coef_lo, _ent_coef_hi, _ent_coef_slope_start, _ppo_settings, _n_real_to_fake_label_flip




import warnings
warnings.filterwarnings("ignore")


_rng = np.random.default_rng()
_ha_column = ["HA"]
_la_column = ["LA.vx","LA.vy","LA.vz","LA.end"]
_feature_column = ["x", "y", "z", "bx", "by", "bz", "tx", "ty", "tz", "end_width"]
_data_path = "data"


np.set_printoptions(suppress=True, precision=3)






# get expert data -----------------------------------------------------------------
def get_single_expert_df(n):
  return pd.read_csv(_data_path+f"/data{n}.csv", skipinitialspace=True)
def get_single_expert_traj(n):
  data = get_single_expert_df(n).astype({_ha_column[0]: int})
  ha = data[_ha_column].to_numpy()
  la = data[_la_column].to_numpy()
  features = data[_feature_column].to_numpy()
  all_obs = data[:][:n_timesteps].to_numpy()

  ha = np.append(np.array([[initialHA]]), ha, axis=0)[:-1]    # gives obs the prev HA instead of current
  la = np.append(np.array([initialLA]), la, axis=0)[:-1]    # gives obs the prev LA instead of current
  obs = np.concatenate((features, la), axis=1)



  dones = np.array([False]*n_timesteps)
  cur_obs = obs[:n_timesteps]
  next_obs = obs[1:n_timesteps+1]


  ha_hidden = np.array([0]*n_timesteps)          # IGNORED IN REWARD NET
  zeros = np.array([[0]*numHA]*n_timesteps)     # IGNORED IN REWARD NET

  ha = np.hstack(ha)
  ha_one_hot = np.eye(numHA)[ha][:n_timesteps]

  obs_with_ha_gt = np.concatenate((cur_obs, ha_one_hot), axis=1)
  obs_with_ha_hidden = np.concatenate((cur_obs, zeros), axis=1)
  next_obs = np.concatenate((next_obs, zeros), axis=1)

  actual_ha = data[_ha_column][:n_timesteps].to_numpy()




  return {
    "obs": obs_with_ha_hidden,
    "next_obs": next_obs,
    "acts": ha_hidden,
    "dones": dones,
    "actual_ha": actual_ha,
    "all_obs": all_obs,
    "gt_obs": obs_with_ha_gt
  }



def evaluate(model, trajectories):
  sum_wrong = 0
  sum_right = 0
  ha_chosen = [0] * numHA
  for i, traj in enumerate(trajectories):
    print(f"DATA ENV {i}")
    last_ha = initialHA
    for all_features, select_features, ha in zip(traj["all_obs"], traj["gt_obs"], traj["actual_ha"]):
      la_actual = all_features[la_idx_wrt_all_obs]
      one_hot = np.zeros(numHA)
      np.put(one_hot,last_ha,1)
      obs = np.concatenate((select_features[:-numHA], one_hot))
      predicted_ha = model.predict(obs)[0]
      print(f"ha, pred1, pred3:             {ha[0]}", end="  ")
      print(f"-1", end="  ")
      print(f"{predicted_ha}")
      sum_wrong += (ha[0]!=predicted_ha)
      sum_right += (ha[0]==predicted_ha)
      last_ha = predicted_ha

      ha_chosen[predicted_ha] += 1

  print()
  print("AVG ACTUAL ERR.               " + str(-1))
  print("AVG PRED ERR 1.               " + str(-1))
  print("AVG PRED ERR 3.               " + str(-1))
  print("ACC 1.                        " + str(-1))
  print("ACC 3.                        " + str(sum_right/(sum_right+sum_wrong)))

  print()
  print("distribution of HA choices")
  print(ha_chosen)
  print()
  return max(ha_chosen)








_traj_train = [get_single_expert_traj(i) for i in range(10)]
_traj_all = [get_single_expert_traj(i) for i in range(30)]

register(id="env-v0", entry_point='custom_envs.envs:Env')
_env_test = gym.make("env-v0")



  

def sanity(model):
  print("SANITY")
  prev_obs = _env_test.reset()
  print(prev_obs)
  for i in range(0,n_timesteps):
    predicted_action = model.predict(prev_obs)[0]
    print(predicted_action)
    cur_state = _env_test.step(predicted_action)
    prev_obs = cur_state[0]
    print([float(f"{num:.3f}") for num in prev_obs])
  print()



_venv = make_vec_env("env-v0", n_envs=training_set, rng=_rng)
for i in range(training_set):
  info = get_single_expert_df(i)
  _venv.env_method("config", info["decMax"][0], info["accMax"][0], info["vMax"][0], info["target"][0], indices=[i])



# GPU ----------------------------------------------------------------------------
import torch

mem_amount = [0]*4
gpus_avail = [4, 5, 6, 7]
for i in range(4):
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
    batch_size=n_timesteps,
    ent_coef=_ppo_settings["ent_coef"],
    learning_rate=_ppo_settings["learning_rate"],
    n_epochs=_ppo_settings["n_epochs"],
    gamma=_ppo_settings["gamma"],
    n_steps=n_timesteps,
    device=device,
    policy_kwargs={
      "net_arch": _policy_net_shape
    }
)
_reward_net = MyRewardNet(
    _venv.observation_space,
    _venv.action_space,
    normalize_input_layer=RunningNorm,
    features_to_use=features_idx,
    la_indices=la_idx_wrt_lim_obs
)
_gail_trainer = GAIL(
    demonstrations=_traj_train,
    demo_batch_size=n_timesteps,                                         # cons
    gen_replay_buffer_capacity=_n_gen_train_steps*_buf_multiplier,      # cons
    gen_train_timesteps=_n_gen_train_steps,                             # gen per round
    n_disc_updates_per_round=_n_disc_updates_per_round,                 # disc per round
    venv=_venv,
    gen_algo=_learner,
    reward_net=_reward_net,
    n_real_to_fake_label_flip=_n_real_to_fake_label_flip
)

evaluate(_learner, _traj_all)
sanity(_learner)
for i in range(_n_train_loops):
    print("LOOP # "+str(i))
    train_info = _gail_trainer.train(_n_gen_train_steps)
    evaluate(_learner, _traj_all)
    sanity(_learner)

