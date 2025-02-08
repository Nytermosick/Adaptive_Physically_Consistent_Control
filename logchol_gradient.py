import pinocchio as pin
import numpy as np
from time import perf_counter

log_cholesky = pin.LogCholeskyParameters(0.2*np.random.randn(10))
In = pin.Inertia.FromLogCholeskyParameters(log_cholesky)
theta_true = log_cholesky.toDynamicParameters()


N = 1000
Y = np.random.randn(N, 10)
y = Y@theta_true + np.random.randn(N)*0.1

params_est = np.zeros(10)
LEARNING_RATE = 0.9/N
MAX_ITER = 10
log_cholesky_estimate_np = np.zeros(10)
theta_est = pin.LogCholeskyParameters(log_cholesky_estimate_np).toDynamicParameters()

total_time_dynamic = 0
for i in range(MAX_ITER):
    t1 = perf_counter()
    theta_est += LEARNING_RATE*Y.T@(y - Y@theta_est)
    t2 = perf_counter()
    total_time_dynamic += t2 - t1

    pseudo = pin.PseudoInertia.FromDynamicParameters(theta_est)
    pseudo_inertia_matrix = np.array(pseudo)
    # print(np.linalg.eigvals(pseudo_inertia_matrix)>0)
    
print('Dynamic parameters estimate:')
print('Original:\n', theta_true)
print('Estimated:\n', theta_est)
print(f'Average time per iteration: {(total_time_dynamic/MAX_ITER)*1000:.3f} ms')

total_time_logchol = 0
for i in range(MAX_ITER):
    t1 = perf_counter()
    log_cholesky_estimate = pin.LogCholeskyParameters(log_cholesky_estimate_np)
    theta_est = log_cholesky_estimate.toDynamicParameters()
    jacobian = log_cholesky.calculateJacobian()
    log_cholesky_estimate_np += LEARNING_RATE*np.linalg.inv(jacobian)@Y.T@(y - Y@theta_est)
    t2 = perf_counter()
    total_time_logchol += t2 - t1

    pseudo = log_cholesky_estimate.toPseudoInertia()
    pseudo_inertia_matrix = np.array(pseudo)
    # print(np.linalg.eigvals(pseudo_inertia_matrix)>0)

print('Log-cholesky estimate:')
print('Original:\n', np.array(log_cholesky))
print('Estimated:\n', log_cholesky_estimate_np)
print(f'Average time per iteration: {(total_time_logchol/MAX_ITER)*1000:.3f} ms')
