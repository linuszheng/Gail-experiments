import numpy as np
import pandas as pd
import re
from scipy.stats import norm


# _data_path = "a-2d-merge-easy/data"
# _traj_path = "trajs/a-best-traj.txt"
# from trajs.a_settings import _n_timesteps, validation_set, training_set, initialLA, pv_stddev, motor_model, la_indices, ha_index

# _data_path = "b-2d-merge-medium/data"
# _traj_path = "trajs/b-best-traj.txt"
# from trajs.b_settings import _n_timesteps, validation_set, training_set, initialLA, pv_stddev, motor_model, la_indices, ha_index

# _data_path = "c-2d-merge-hard/data"
# _traj_path = "trajs/c-best-traj.txt"
# from trajs.c_settings import _n_timesteps, validation_set, training_set, initialLA, pv_stddev, motor_model, la_indices, ha_index

# _data_path = "d-2d-merge-impossible/data"
# _traj_path = "trajs/d-best-traj.txt"
# from trajs.d_settings import _n_timesteps, validation_set, training_set, initialLA, pv_stddev, motor_model, la_indices, ha_index

# _data_path = "f-2d-highway/data"
# _traj_path = "trajs/f-best-traj.txt"
# from trajs.f_settings import _n_timesteps, validation_set, training_set, initialLA, pv_stddev, motor_model, la_indices, ha_index



# get expert data -----------------------------------------------------------------
def get_single_expert_df(n):
  return pd.read_csv(_data_path+f"/data{n}.csv", skipinitialspace=True)
def get_single_expert_traj(n):
  data = get_single_expert_df(n)
  all_obs = data[:][:_n_timesteps].to_numpy()
  return all_obs




# get pred data -------------------
def get_all_pred_ha():
  f = open(_traj_path)
  res = []
  cur = []
  n = 0
  f.readline()
  while True:
    line = f.readline()
    if ha := re.findall("ha, pred1, pred3: \s+ \d  \d  (\d)", line):
      ha = int(ha[0])
      cur.append(ha)
    elif not line:
      res.append(cur)
      break
    else:
      res.append(cur)
      cur = []
      n += 1
  return res



def get_err(a_, b_, stdev_):
  # res = [np.log(norm.pdf((a-b)/stdev)) for a, b, stdev in zip(a_, b_, stdev_)]
  res = [norm(a, stdev).logpdf(b) for a, b, stdev in zip(a_, b_, stdev_)]
  return sum(res)


def evaluate(single_expert_traj, single_pred_ha):
  cum_err = 0
  cum_acc = 0
  last_la = initialLA
  for i in range(_n_timesteps):
    info = single_expert_traj[i]
    pred_ha = single_pred_ha[i]
    actual_la = info[la_indices]
    pred_la = motor_model(pred_ha, info, last_la)
    err = get_err(actual_la, pred_la, pv_stddev)
    cum_err += err
    cum_acc += (info[ha_index]==pred_ha)
    last_la = actual_la
  return (cum_err / _n_timesteps, cum_acc / _n_timesteps)



all_pred_ha = get_all_pred_ha()
all_expert_traj = [get_single_expert_traj(i) for i in range(validation_set)]
sum_err = 0
sum_acc = 0
for i in range(training_set, validation_set):
  eval_res = evaluate(all_expert_traj[i], all_pred_ha[i])
  sum_err += eval_res[0]
  sum_acc += eval_res[1]

set_size = validation_set - training_set
print("experiment " + _data_path)
print("avg ll")
print(sum_err / set_size)
print("avg acc")
print(sum_acc / set_size)

