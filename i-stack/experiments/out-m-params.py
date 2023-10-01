
_max_disc_acc_until_quit = 2.0
_max_mode_until_quit = 2.0
def _learning_rate_func(progress):
  lr_start = .0008
  lr_end = .0002
  lr_diff = lr_end - lr_start
  return lr_start + progress * lr_diff
_n_gen_train_steps = 50
_n_disc_updates_per_round = 3
_buf_multiplier = 3
_policy_net_shape = dict(pi=[16, 16, 16], vf=[16, 16, 16])
_ent_coef_lo = .009
_ent_coef_hi = .0015
_ent_coef_slope_start = .8
_ppo_settings = {
  "ent_coef": _ent_coef_lo,
  "learning_rate": _learning_rate_func,
  "n_epochs": 50,
  "gamma": .99,
}
_n_real_to_fake_label_flip = 0