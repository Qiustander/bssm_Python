import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from Models.ssm_nlg import NonlinearSSM
from Models.check_argument import *
from sklearn.metrics import mean_squared_error
from Inference.Kalman.extended_kalman_smoother import extended_kalman_smoother
from Inference.Kalman.extended_kalman_filter import extended_kalman_filter
import matplotlib.pyplot as plt

def debug_plot(tfp_result, r_result, true_state):
    tfp, = plt.plot(tfp_result, color='blue', linewidth=1)
    r, = plt.plot(r_result, color='green', linewidth=1)
    true, = plt.plot(true_state, '-.', color='red', linewidth=1)
    plt.legend(handles=[tfp, r, true], labels=['smoother', 'filter', 'true'])

    plt.show()
    print(f'MSE error of Smoother: {mean_squared_error(true_state, tfp_result)}')
    print(f'MSE error of Filter: {mean_squared_error(true_state, r_result)}')

ro.r("""
set.seed(1)
n <- 200 
sigma_y <- 0.1
sigma_x <- c(0.1, 0.1, 0.2, 0.1)
a1 <- c(0, 1, 0.5, 0)
P1 <- c(1, 2, 1, 1.5)
x <- matrix(0, nrow=n, ncol=4)
y <- matrix(0, nrow=n, ncol=3)
dt <- 0.3

x[1, ] <- rnorm(4, a1, diag(P1))
y[1, ] <- rnorm(3,
    c(x[1, 1]**2, x[1, 2]**3, 0.5*x[1, 3]+2*x[1, 4]+x[1, 1]+x[1, 2]), sigma_y)
known_params <- c(dT = dt, 
                  a11 = a1[1], a12 = a1[2], a11 = a1[3], a12 = a1[4], 
                  P11 = P1[1], P12 = P1[2], P11 = P1[3], P12 = P1[4])
for(i in 2:n) {
    x[i, ] <- rnorm(4, c(0.8*x[i-1, 1] + dt*x[i-1, 2], 
                         0.7*x[i-1, 2] + dt*x[i-1, 3], 
                         0.6*x[i-1, 3] + dt*x[i-1, 4],
                         0.6*x[i-1, 4] + dt*x[i-1, 1]), 
                    sigma_x)
    y[i, ] <- rnorm(3, c(x[i, 1]**2, x[i, 2]**3, 0.5*x[i, 3]+2*x[1, 4]+x[i, 1]+x[i, 2]), 
                         sigma_y)
}

    """)

observation = np.array(ro.r("y"))
size_y, observation = check_y(observation.astype("float32"))  # return time length and feature numbers
num_timesteps, observation_size = size_y

model_obj = NonlinearSSM.create_model(num_timesteps=num_timesteps,
                                      observation_size=observation_size,
                                      latent_size=4,
                                      initial_state_mean=np.array([0., 1., 0.5, 0.]),
                                      initial_state_cov=np.diag([1, 2, 1, 1.5]),
                                      state_noise_std=np.diag([0.1, 0.1, 0.2, 0.1]),
                                      obs_noise_std=np.diag([0.1, 0.1, 0.1]),
                                      dt=0.3,
                                      nonlinear_type="nlg_mv_model")

infer_result_ekf = extended_kalman_filter(model_obj, observation)
infer_result = extended_kalman_smoother(model_obj, observation)
for i in range(4):
    debug_plot(infer_result[0][:,i].numpy(), infer_result_ekf[0][:,i], np.array(ro.r("x"))[:,i] )


