import numpy as np

_min_performance_to_save = .60


training_set = 10
validation_set = 30 # including training_set


feature_indices = [0, 2, 5, 10, 15]

pred_var = ["LA.steer", "LA.acc"]
pv_range = [
    [-0.3, 0.3],
    [-30, 30]
]
pv_stddev = [0.01, 2.0]

numHA = 4
_n_timesteps = 149
initialHA = 0
initialLA = [0,0]

KP_H = 0.5 # Turning rate
TURN_HEADING = 0.15 # Target heading when turning
TURN_TARGET = 30 # How much to adjust when targeting a lane (higher = smoother)
max_velocity = 40 # Maximum velocity
turn_velocity = 30 # Turning velocity

lane_diff = 4
def laneFinder(y):
    return round(y / lane_diff)

def motor_model(ha, data, data_prev):
    target_acc = 0.0
    target_heading = 0.0
    if ha == 0:
        target_acc = max_velocity - data[2]

        target_y = laneFinder(data[1]) * 4
        target_heading = np.arctan((target_y - data[1]) / TURN_TARGET)
    elif ha == 1:
        target_acc = data[12] - data[2]

        target_y = laneFinder(data[1]) * 4
        target_heading = np.arctan((target_y - data[1]) / TURN_TARGET)
    elif ha == 2:
        target_acc = turn_velocity - data[2]
        target_heading = -TURN_HEADING
    else:
        target_acc = turn_velocity - data[2]
        target_heading = TURN_HEADING

    target_steer = target_heading - data[4]
    if(target_steer > data_prev[0]):
        target_steer = min(target_steer, data_prev[0] + 0.04)
    else:
        target_steer = max(target_steer, data_prev[0] - 0.04)

    if(target_acc > data_prev[1]):  
        target_acc = min(target_acc, data_prev[1] + 4)
    else:
        target_acc = max(target_acc, data_prev[1] - 6)

    return [target_steer, target_acc]
