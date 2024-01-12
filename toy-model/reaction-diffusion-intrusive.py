import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import matplotlib.pyplot as plt
from Toy_model import *
import sys 
sys.path.append("../utils") 
from Evaluation_plot import *

toymodel = Toymodel()

# Build Training data
dlt_t = 0.0001
data_x, data_y, data_lace, _ = toymodel.genarate_training_data(steps = 10, traj_num = 500, dlt_t = dlt_t)

mu_intrusive, lam_intrusive, err_history_intrusive = optimize_with_gradient_descent_tf_small_batch(data_x, data_y, data_lace, 10, 0.0001, 2000, batch_size = 512)

np.save('plot_data/mu_intrusive.npy', mu_intrusive)
np.save('plot_data/lam_intrusive.npy', lam_intrusive)
np.save('plot_data/err_history_intrusive.npy', err_history_intrusive)