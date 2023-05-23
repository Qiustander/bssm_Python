import pytest
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from Models.ssm_nlg import NonlinearSSM
from Models.check_argument import *
import os.path as pth
import os
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# gpu = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(device=gpu[0], enable=True)

# Automatic convertion between R and Python objects
numpy2ri.activate()

# Must call the function via the package, if the desired function is
# linked with other functions in the package. The simple call via ro.r("""source""")
# will only create a simple object that miss the link.
base = importr('base', lib_loc="/usr/lib/R/library")
bssm = importr('bssm', lib_loc=f"{pth.expanduser('~')}/R/x86_64-pc-linux-gnu-library/4.3")
stat = importr('stats', lib_loc="/usr/lib/R/library")


class TestExtendedKalmanFilter:
    """
    Test Extended Kalman Filter - ssm_nlg
    """

    def test_kffilter_TFP_arexp(self):
        ro.r("""
        mu <- -0.2
        rho <- 0.7
        n <- 150
        sigma_y <- 0.1
        sigma_x <- 1
        x <- numeric(n)
        x[1] <- rnorm(1, mu, sigma_x / sqrt(1 - rho^2))
        for(i in 2:length(x)) {
          x[i] <- rnorm(1, mu * (1 - rho) + rho * x[i - 1], sigma_x)
        }
        y <- rnorm(n, exp(x), sigma_y)
        
        pntrs <- cpp_example_model("nlg_ar_exp")
        
        model_nlg <- ssm_nlg(y = y, a1 = pntrs$a1, P1 = pntrs$P1,
          Z = pntrs$Z_fn, H = pntrs$H_fn, T = pntrs$T_fn, R = pntrs$R_fn,
          Z_gn = pntrs$Z_gn, T_gn = pntrs$T_gn,
          theta = c(mu= mu, rho = rho,
            log_sigma_x = log(sigma_x), log_sigma_y = log(sigma_y)),
          log_prior_pdf = pntrs$log_prior_pdf,
          n_states = 1, n_etas = 1, state_names = "state")
        
        infer_result <- ekf(model_nlg, iekf_iter = 0)
            """)
        r_result = ro.r("infer_result")
        observation = np.array(ro.r("y"))
        size_y, observation = check_y(observation.astype("float32")) # return time length and feature numbers
        num_timesteps, observation_size = size_y

        model_obj = NonlinearSSM.create_model(num_timesteps=num_timesteps,
                                 observation_size=observation_size,
                                 latent_size=1,
                                 initial_state_mean=0.1,
                                 initial_state_cov=0,
                                mu_state=0.2,
                                rho_state=0.7,
                                 state_noise_std=1.,
                                 obs_noise_std=0.1,
                                 nonlinear_type="nlg_ar_exp")

        infer_result = model_obj.extended_Kalman_filter(observation)

        # true_state = np.array(ro.r("x"))[..., None]
        # plt.plot(infer_result[0].numpy(), color='blue', linewidth=1)
        # plt.plot(r_result[1], color='green', linewidth=1)
        # plt.plot(true_state, '-.', color='red', linewidth=1)
        # plt.show()
        # compare loglik
        tf.debugging.assert_near(r_result[-1], infer_result[-2].numpy().sum(), atol=1e-2)
        # compare filtered_means
        tf.debugging.assert_near(r_result[1], infer_result[0].numpy(), atol=1e-1)
        # compare filtered_covs
        tf.debugging.assert_near(r_result[3], infer_result[1].numpy().transpose(1, 2, 0), atol=1e-1)
        # compare predicted_means
        #TODO: in r it computes the next step prediction! but in ekf does not
        tf.debugging.assert_near(r_result[0][:-1, ...], infer_result[2].numpy(), atol=1e-1)
        # compare predicted_covs
        tf.debugging.assert_near(r_result[2][..., :-1], infer_result[3].numpy().transpose(1, 2, 0), atol=1e-1)

    def test_kffilter_TFP_sinexp(self):
        ro.r("""
        n <- 150
        x <- y <- numeric(n) + 0.1
        y[1] <- rnorm(1, exp(x[1]), 0.1)
        for(i in 1:(n-1)) {
         x[i+1] <- rnorm(1, sin(x[i]), 0.1)
         y[i+1] <- rnorm(1, exp(x[i+1]), 0.2)
        }

        pntrs <- cpp_example_model("nlg_sin_exp")

        model_nlg <- ssm_nlg(y = y, a1 = pntrs$a1, P1 = pntrs$P1,
          Z = pntrs$Z_fn, H = pntrs$H_fn, T = pntrs$T_fn, R = pntrs$R_fn,
          Z_gn = pntrs$Z_gn, T_gn = pntrs$T_gn,
          theta = c(log_H = log(0.2), log_R = log(0.1)),
          log_prior_pdf = pntrs$log_prior_pdf,
          n_states = 1, n_etas = 1, state_names = "state")

        infer_result <- ekf(model_nlg, iekf_iter = 0)
            """)
        r_result = ro.r("infer_result")

        observation = np.array(ro.r("y"))
        size_y, observation = check_y(observation.astype("float32"))  # return time length and feature numbers
        num_timesteps, observation_size = size_y

        model_obj = NonlinearSSM.create_model(num_timesteps=num_timesteps,
                                             observation_size=observation_size,
                                             latent_size=1,
                                             initial_state_mean=0,
                                             initial_state_cov=1.,
                                             state_noise_std=0.1,
                                             obs_noise_std=0.2,
                                             nonlinear_type="nlg_sin_exp")
        infer_result = model_obj.extended_Kalman_filter(observation)

        # compare loglik
        tf.debugging.assert_near(r_result[-1], infer_result[-2].numpy().sum(), atol=1e-2)
        # compare filtered_means
        tf.debugging.assert_near(r_result[1], infer_result[0].numpy(), atol=1e-4)
        # compare filtered_covs
        tf.debugging.assert_near(r_result[3], infer_result[1].numpy().transpose(1, 2, 0), atol=1e-4)
        # compare predicted_means
        #TODO: in r it computes the next step prediction! but in ekf does not
        tf.debugging.assert_near(r_result[0][:-1, ...], infer_result[2].numpy(), atol=1e-4)
        # compare predicted_covs
        tf.debugging.assert_near(r_result[2][..., :-1], infer_result[3].numpy().transpose(1, 2, 0), atol=1e-1)

        # true_state = np.array(ro.r("x"))[..., None]
        # plt.plot(infer_result[0].numpy(), color='blue', linewidth=1)
        # plt.plot(r_result[1], color='green', linewidth=1)
        # plt.plot(true_state, '-.', color='red', linewidth=1)
        # plt.show()