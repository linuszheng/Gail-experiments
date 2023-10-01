# Setting: 1D-target


training_set = 5
validation_set = 20

n_timesteps = 49

pv_range = [
    [-2, 2],
    [-2, 2],
    [-2, 2],
    [-2, 2],
]
pv_stddev = [0.2, 0.2, 0.2, 0.2]


initialHA = 0
initialLA = [0, 0, 0, 0]


numHA = 2


la_idx_wrt_lim_obs = [10, 11, 12, 13]
la_idx_wrt_all_obs = [10, 11, 12, 13]
features_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]




def motor_model(ha, data, data_prev):
    bx, by, bz = data[3] - data[0], data[4] - data[1], data[5] - data[2]
    
    if ha == 0:
        return [bx * 5.0, by * 5.0, bz * 5.0, 0.6]
    
    if data_prev[13] >= 0.5:
        return [bx * 5.0, by * 5.0, bz * 5.0, 0.3]
    elif data_prev[13] >= 0.3:
        return [bx * 5.0, by * 5.0, bz * 5.0, 0]
    elif data_prev[13] >= 0.1:
        return [bx * 5.0, by * 5.0, bz * 5.0, -0.3]
    elif data_prev[13] >= -0.1:
        return [bx * 5.0, by * 5.0, bz * 5.0, -0.6]
    
    vx = 5 * (data[6] - data[0])
    vy = 5 * (data[7] - data[1])
    vz = 5 * (data[8] - data[2])
    end = -0.6
    
    return [vx, vy, vz, end]
