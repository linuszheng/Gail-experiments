





_n_gen_train_steps = 50
_n_disc_updates_per_round = 3
_buf_multiplier = 2
_policy_net_shape = dict(pi=[16, 16, 16], vf=[16, 16, 16])
_ppo_settings = {
  "ent_coef": 0.0005,
  "learning_rate": 0.0005,
  "n_epochs": 30,
  "gamma": 1,
}

# RESULT: PEAK ACCURACY 68% THEN MODE COLLAPSE