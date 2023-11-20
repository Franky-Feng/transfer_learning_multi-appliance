import numpy as np
import matplotlib.pyplot as plt
import Arguments

appliance = ['kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine']
para = params_appliance = Arguments.refit_params_appliance
aggregate_mean = params_appliance['aggregate']['mean']
aggregate_std = params_appliance['aggregate']['std']
save_path = 'total/'

file_path = 'total/test_set.npy'
test_set = np.load(file_path)


half_length = len(test_set) // 10
subset = test_set[:half_length]
np.save(save_path + 'test_set_small.npy', subset)

