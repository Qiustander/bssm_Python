import pytest
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from Models.ssm_lg import LinearGaussianSSM
from Models.check_argument import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os.path as pth
import tensorflow as tf

bssm = importr('bssm', lib_loc=f"{pth.expanduser('~')}/R/x86_64-pc-linux-gnu-library/4.3")


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
n <- 100
    x1 <- rnorm(n)
    x2 <- rnorm(n)
    b1 <- 1 + cumsum(rnorm(n, sd = 0.5))
    b2 <- 2 + cumsum(rnorm(n, sd = 0.1))
    y <- 1 + b1 * x1 + b2 * x2 + rnorm(n, sd = 0.1)

    Z <- rbind(1, x1, x2)
    H <- 0.1
    T <- diag(3)
    R <- diag(c(0, 1, 0.1))
    a1 <- rep(0, 3)
    P1 <- diag(10, 3)

    # updates the model given the current values of the parameters
    update_fn <- function(theta) {
      R <- diag(c(0, theta[1], theta[2]))
      dim(R) <- c(3, 3, 1)
      list(R = R, H = theta[3])
    }
    # prior for standard deviations as half-normal(1)
    prior_fn <- function(theta) {
      if(any(theta < 0)) {
        log_p <- -Inf
      } else {
        log_p <- sum(dnorm(theta, 0, 1, log = TRUE))
      }
      log_p
    }

    model_r <- ssm_ulg(y, Z, H, T, R, a1, P1,
             init_theta = c(1, 0.1, 0.1),
             update_fn = update_fn, prior_fn = prior_fn,
             state_names = c("level", "b1", "b2"),
             # using default values, but being explicit for testing purposes
             C = matrix(0, 3, 1), D = numeric(1))

    """)

y = np.array(ro.r["y"])
obs_mtx = np.array(ro.r("Z"))
obs_mtx_noise = np.array(ro.r("H"))
state_mtx= np.array(ro.r("T"))
state_mtx_noise=np.array(ro.r("R"))
prior_mean = np.array(ro.r("a1"))
prior_cov = np.array(ro.r("P1"))
input_state = ro.r("""matrix(0, 3, 1)""")
input_obs =  np.array([0.])

size_y, observation = check_y(y)
num_timesteps, observation_size = size_y
model_obj = LinearGaussianSSM.create_model(num_timesteps=num_timesteps,
                                           observation_size=observation_size,
                                           latent_size=3,
                                           initial_state_mean=prior_mean,
                                           initial_state_cov=prior_cov,
                                           state_noise_std=state_mtx_noise,
                                           obs_noise_std=obs_mtx_noise,
                                           obs_mtx=obs_mtx,
                                           state_mtx=state_mtx)
infer_result_kf = model_obj.forward_filter(tf.convert_to_tensor(observation, dtype=model_obj.dtype))
infer_result_smoother = model_obj.posterior_marginals(tf.convert_to_tensor(observation, dtype=model_obj.dtype))

for i in range(3):
    debug_plot(infer_result_smoother[0][:,i].numpy(), infer_result_kf.filtered_means[:,i], np.array(ro.r("b1")) )


