# Setting: 1D-target

_min_performance_to_save = .87


training_set = 10
validation_set = 30

_n_timesteps = 125

pv_range = [[-50, 50]]
pv_stddev = [0.5]


initialHA = 0
initialLA = [0]


numHA = 3


la_idx_wrt_lim_obs = [6]
la_idx_wrt_all_obs = [1]
features_idx = [0, 1, 2, 3, 4, 5]

la_indices = [1]
ha_index = [8]


def motor_model(ha, data, data_prev):
    if ha == 0:
        return [min(data_prev[0] + 1, data[4])]
    elif ha == 1:
        if data_prev[0] < 0:
            return [min(data_prev[0] + 1, data[4])]
        elif data_prev[0] > 0:
            return [max(data_prev[0] - 1, data[3])]
    else:
        return [max(data_prev[0] - 1, data[3])]
