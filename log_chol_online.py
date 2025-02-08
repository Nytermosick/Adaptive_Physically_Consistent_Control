import pinocchio as pin
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt

log_cholesky = pin.LogCholeskyParameters(0.5*np.random.randn(10))
In = pin.Inertia.FromLogCholeskyParameters(log_cholesky)
theta_true = log_cholesky.toDynamicParameters()
# theta in physical consistancy
# eta in R^10 
# y = Y@theta(eta)

# theta_optimal = argmin(y - Y@theta)
# subject to theta_optimal in Physical Consistancy

# eta_optimal = min(y - Y@theta(eta))

# dtheta_hat/dt = -gamma*Y*s
# # POSIBLY BROKE CONSTRAINT
# dtheta/deta = Jac

# deta_hat/dt = -Jac*gamma*Y*s

# deta_hat/dt = (Jac)^{-1}*dtheta_hat/dt
#= -Jac*gamma*Y*s

N = 2000
Y = np.random.randn(N, 10)
y = Y@theta_true + np.random.randn(N)*0.01

params_est = np.zeros(10)
BATCH_SIZE = 40
LEARNING_RATE = 0.01/BATCH_SIZE
MAX_ITER = 10
log_cholesky_estimate_np = np.zeros(10)
theta_est = pin.LogCholeskyParameters(log_cholesky_estimate_np).toDynamicParameters()

total_time_logchol = 0
history = []
true_params = np.array(log_cholesky)

for i in range(N):
    # Get last BATCH_SIZE points for batch learning
    t1 = perf_counter()
    start_idx = max(0, i-BATCH_SIZE+1)
    Y_batch = Y[start_idx:i+1]
    y_batch = y[start_idx:i+1]
    log_cholesky_estimate = pin.LogCholeskyParameters(log_cholesky_estimate_np)
    theta_est = log_cholesky_estimate.toDynamicParameters()
    jacobian = log_cholesky.calculateJacobian()
    log_cholesky_estimate_np += LEARNING_RATE*np.linalg.inv(jacobian)@Y_batch.T@(y_batch - Y_batch@theta_est)
    t2 = perf_counter()
    total_time_logchol += t2 - t1

    pseudo = log_cholesky_estimate.toPseudoInertia()
    pseudo_inertia_matrix = np.array(pseudo)
    # print(np.linalg.eigvals(pseudo_inertia_matrix)>0)
    
    # Store history of parameter estimates
    history.append(log_cholesky_estimate_np.copy())

print('Log-cholesky estimate:')
print('Original:\n', np.array(log_cholesky))
print('Estimated:\n', log_cholesky_estimate_np)
print(f'Average time per iteration: {(total_time_logchol/N)*1000:.3f} ms')

# Plot learning history
history = np.array(history)
plt.figure(figsize=(10, 6))
for j in range(10):
    plt.plot(history[:, j], label=r'$\eta_{' + f'{j+1}' +r'}$')
    plt.axhline(y=true_params[j], color=plt.gca().lines[-1].get_color(), linestyle='--', alpha=0.5)
plt.xlabel('Iteration')
plt.ylabel('Parameter Value')
plt.title('Learning History of Log-Cholesky Parameters')
plt.legend()
plt.grid(True)
plt.show()
