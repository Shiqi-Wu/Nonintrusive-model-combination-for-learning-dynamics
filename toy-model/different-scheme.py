import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from Toy_model import *
import sys 
sys.path.append("../utils") 
from Evaluation_plot import *

toymodel = Toymodel()

dlt_t = 0.0001
steps = 1000
u0 = 2 * np.random.rand(toymodel.dim - 1) - 1
traj_euler = toymodel.generate_traj(steps, u0, dlt_t = dlt_t)
traj_ivp = toymodel.generate_traj_solve_ivp(steps, u0, dlt_t = dlt_t)

fig, axs = plt.subplots(toymodel.dim - 1, 1, figsize=(10, 2 * (toymodel.dim - 1)))
for i in range(toymodel.dim - 1):
    axs[i].plot(traj_euler[:, i], label='Euler Method')
    axs[i].plot(traj_ivp[:, i], label='solve_ivp Method', linestyle='--')
    axs[i].set_title(f'Dimension {i+1}')
    axs[i].legend()

plt.tight_layout()
plt.show()

# x_data, y_data, x_laplace_data, y_laplace_data = toymodel.genarate_training_data(steps = 10, traj_num = 500, dlt_t = dlt_t, type = 'Euler-forward')
x_data, y_data, x_laplace_data, y_laplace_data = toymodel.genarate_training_data(steps = 10, traj_num = 500, dlt_t = dlt_t, type = 'Solve_ivp')

reaction_term = toymodel.reaction_term(x_data)

iterative_steps = 10

combined_model_euler_forward = LinearModelCombiningNN()
err_history_euler_forward = combined_model_euler_forward.iterative_train(x_data, y_data, 
                        x_laplace_data, y_laplace_data, 
                        iterative_steps = iterative_steps, epochs = 200, type = 'Euler-forward', 
                        acc = False)

combined_model_euler_backward = LinearModelCombiningNN()
err_history_euler_backward = combined_model_euler_backward.iterative_train(x_data, y_data, 
                        x_laplace_data, y_laplace_data, 
                        iterative_steps = iterative_steps, epochs = 200, type = 'Euler-backward', 
                        acc = False)

combined_model_central_difference = LinearModelCombiningNN()
err_history_central_difference = combined_model_central_difference.iterative_train(x_data, y_data, 
                        x_laplace_data, y_laplace_data, 
                        iterative_steps = iterative_steps, epochs = 200, type = 'Central-difference', 
                        acc = False)

predict_num = 10
steps = 1000
dlt_x = 0.05

euler_forward_history = []
euler_backward_history = []
central_difference_history = []

euler_forward_traj_history = []
euler_backward_traj_history = []
central_difference_traj_history = []

test_traj_history = []
for i in range(predict_num):
    u0 = np.random.rand(toymodel.dim - 1)
    u_test = toymodel.generate_traj_solve_ivp(steps = steps, u0 = u0, dlt_t = dlt_t)
    u_euler_forward = combined_model_euler_forward.euler_forward_prediction(u0,steps)
    u_euler_backward = combined_model_euler_backward.euler_backward_prediction(u0, steps)
    u_central_difference = combined_model_central_difference.central_difference_prediction(u0, steps)

    euler_forward_history.append(np.linalg.norm(u_test - u_euler_forward, ord=2, axis = 1)/np.linalg.norm(u_test, ord = 2))
    euler_backward_history.append(np.linalg.norm(u_test - u_euler_backward, ord=2, axis = 1)/np.linalg.norm(u_test, ord = 2))
    central_difference_history.append(np.linalg.norm(u_test - u_central_difference, ord=2, axis = 1)/np.linalg.norm(u_test, ord = 2))

    euler_forward_traj_history.append(u_euler_forward)
    euler_backward_traj_history.append(u_euler_backward)
    central_difference_traj_history.append(u_central_difference)
    test_traj_history.append(u_test)
    print(i)

import numpy as np
import os

save_dir = 'plot_data'
os.makedirs(save_dir, exist_ok=True)

# Save the lists
np.save(os.path.join(save_dir, 'euler_forward_history.npy'), euler_forward_history)
np.save(os.path.join(save_dir, 'euler_backward_history.npy'), euler_backward_history)
np.save(os.path.join(save_dir, 'central_difference_history.npy'), central_difference_history)
np.save(os.path.join(save_dir, 'euler_forward_traj_history.npy'), euler_forward_traj_history)
np.save(os.path.join(save_dir, 'euler_backward_traj_history.npy'), euler_backward_traj_history)
np.save(os.path.join(save_dir, 'central_difference_traj_history.npy'), central_difference_traj_history)
np.save(os.path.join(save_dir, 'test_traj_history.npy'), test_traj_history)

print("Experiments are done.")