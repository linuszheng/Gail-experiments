def _learning_rate_func(progress):
  lr_start = .0005
  lr_end = .0001
  lr_diff = lr_end - lr_start
  return lr_start + progress * lr_diff
  
_n_gen_train_steps = 50
_n_disc_updates_per_round = 3
_buf_multiplier = 2
_policy_net_shape = dict(pi=[16, 16, 16], vf=[16, 16, 16])
_ppo_settings = {
  "ent_coef": 0.0005,
  "learning_rate": _learning_rate_func,
  "n_epochs": 30,
  "gamma": 1,
}