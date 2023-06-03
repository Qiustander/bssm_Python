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
        a1 <- rep(0, 6)
        P1 <- rep (1000, 6)
        n <- 100
        sd_x <- c(1,1,0.02)
        sd_y <- c(1,1,1)
        T <- c(1,0,0,1,0,0,
               0,1,0,0,1,0,
               0,0,1,0,0,1,
               0,0,0,1,0,0,
               0,0,0,0,1,0,
               0,0,0,0,0,1)
        T <- matrix(T,ncol=6,nrow=6,byrow=T)
        Z <- c(1,0,0,0,0,0,
               0,1,0,0,0,0,
               0,0,1,0,0,0)
        Z <- matrix(Z,ncol=6,nrow=3, byrow=T)
        R <- c(0.5*sd_x[1], 0, 0,
               0, 0.5*sd_x[2], 0,
               0, 0, 0.5*sd_x[3],
               sd_x[1], 0, 0,
               0, sd_x[2], 0,
               0, 0, sd_x[3])
        R <- matrix(R,ncol=3,nrow=6,byrow=T)
        H <- diag(sd_y)
        x <- matrix(0, nrow=n, ncol=6) # state
        y <- matrix(0, nrow=n, ncol=3) # observation
        x[1, ] <- rnorm(6, a1, P1)
        y[1, ] <- rnorm(3, Z%*%x[1, ], sd_y)
        for(i in 2:n) {
            x[i, ] <- T%*% x[i-1, ]+ R%*%rnorm(3, 0, 1)
            y[i, ] <- rnorm(3, Z%*%x[i, ], sd_y)
        }
        model_r <- ssm_mlg(y, H = H,
                           R = R, Z = Z, T = T, P1 = diag(P1))

    """)

y = np.array(ro.r["y"])
obs_mtx = np.array(ro.r("model_r$Z"))
obs_mtx_noise = np.array(ro.r("model_r$H"))
state_mtx= np.array(ro.r("model_r$T"))
state_mtx_noise=np.array(ro.r("model_r$R"))
prior_mean = np.array(ro.r("model_r$a1"))
prior_cov = np.array(ro.r("model_r$P1"))
input_state = ro.r("model_r$C")
input_obs =  ro.r("model_r$D")

size_y, observation = check_y(y)
num_timesteps, observation_size = size_y
model_obj = LinearGaussianSSM.create_model(num_timesteps=num_timesteps,
                                           observation_size=observation_size,
                                           latent_size=6,
                                           initial_state_mean=prior_mean,
                                           initial_state_cov=prior_cov,
                                           state_noise_std=state_mtx_noise,
                                           obs_noise_std=obs_mtx_noise,
                                           obs_mtx=obs_mtx,
                                           state_mtx=state_mtx)
infer_result_kf = model_obj.forward_filter(tf.convert_to_tensor(observation, dtype=model_obj.dtype))
infer_result_smoother = model_obj.posterior_marginals(tf.convert_to_tensor(observation, dtype=model_obj.dtype))

for i in range(6):
    debug_plot(infer_result_smoother[0][10:,i].numpy(), infer_result_kf.filtered_means[10:,i], np.array(ro.r("x"))[10:,i] )


