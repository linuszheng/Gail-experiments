# Setting: 1D-target


training_set = 10
validation_set = 30


pv_range = [[-50, 50]]
pv_stddev = [0.5]


initialHA = 0
initialLA = [0]


numHA = 3


la_idex_wrt_lim_obs = [6]
features_idx = [0, 1, 2, 3, 4, 5]


def motor_model(ha, data, data_prev):
    if ha == 0:
        return [min(data_prev[6] + 1, data[2])]
    elif ha == 1:
        if data_prev[0] < 0:
            return [min(data_prev[6] + 1, data[2])]
        elif data_prev[0] > 0:
            return [max(data_prev[6] - 1, data[1])]
    else:
        return [max(data_prev[6] - 1, data[1])]