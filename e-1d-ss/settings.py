# Setting: 1D-target


training_set = 10
validation_set = 30 # including training_set
train_time = 10000
patience = 500
sim_time = 126
samples = 100
pred_var = ["LA.acc"]
pv_range = [[-50, 50]]
pv_stddev = [0.5]

numHA = 3
def motor_model(ha, data, data_prev):
    if ha == 0:
        return [min(data_prev[0] + 1, data["accMax"])]
    elif ha == 1:
        if data_prev[0] < 0:
            return [min(data_prev[0] + 1, data["accMax"])]
        elif data_prev[0] > 0:
            return [max(data_prev[0] - 1, data["decMax"])]
    else:
        return [max(data_prev[0] - 1, data["decMax"])]
