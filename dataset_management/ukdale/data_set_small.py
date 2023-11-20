import numpy as np
import matplotlib.pyplot as plt
import Arguments

appliance = ['kettle', 'microwave', 'fridge', 'dishwasher', 'washingmachine']
save_path = 'total/'

file_path = 'total/val_set.npy'
test_set = np.load(file_path)


half_length = len(test_set) // 10
subset = test_set[:half_length]
np.save(save_path + 'val_set_small.npy', subset)

