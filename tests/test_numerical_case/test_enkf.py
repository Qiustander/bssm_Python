import pytest
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from Models.ssm_nlg import NonlinearSSM
from Inference.Kalman.ensemble_kalman_filter import ensemble_kalman_filter
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


class TestEnsembleKalmanFilter:
    """
    Test Ensenmble Kalman Filter - ssm_nlg
    """

    def test_enkffilter_TFP_arexp(self):
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

        infer_result = model_obj.ensemble_Kalman_filter(observation, num_particles=20)
        infer_result_standalone = ensemble_kalman_filter(model_obj, observation, num_particles=20)

        true_state = np.array(ro.r("x"))[..., None]
        # plt.plot(infer_result[0].numpy(), color='blue', linewidth=1)
        plt.plot(infer_result_standalone[0].numpy(), color='black', linewidth=1)
        plt.plot(r_result[1], color='green', linewidth=1)
        plt.plot(true_state, '-.', color='red', linewidth=1)
        plt.show()
        print('MSE error of the EKF from bssm: %.4f' % np.sum((r_result[1] - true_state) ** 2))
        print('MSE error of the EnKF from TFP: %.4f' % np.sum((infer_result[0].numpy() - true_state) ** 2))

    def test_enkffilter_TFP_sinexp(self):
        ro.r("""
        n <- 150
        x <- y <- numeric(n) + 0.1
        y[1] <- rnorm(1, exp(x[1]), 0.2)
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
        true_state = np.array(ro.r("x"))[..., None]
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
        infer_result = model_obj.ensemble_Kalman_filter(observation, num_particles=20)
        infer_result_standalone = ensemble_kalman_filter(model_obj, observation, num_particles=20)

        print('MSE error of the EKF from bssm: %.4f' % np.sum((r_result[1] - true_state) ** 2))
        print('MSE error of the EnKF from TFP: %.4f' % np.sum((infer_result[0].numpy() - true_state) ** 2))

        # true_state = np.array(ro.r("x"))[..., None]
        # plt.plot(infer_result[0].numpy(), color='blue', linewidth=1)
        # plt.plot(r_result[1], color='green', linewidth=1)
        # plt.plot(true_state, '-.', color='red', linewidth=1)
        # plt.show()

    def test_kffilter_TFP_mvmodel(self):
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

        Rcpp::sourceCpp("ssm_nlg_mv_model.cpp")
        pntrs <- create_xptrs()

        model_nlg <- ssm_nlg(y = y, a1 = pntrs$a1, P1 = pntrs$P1,
          Z = pntrs$Z_fn, H = pntrs$H_fn, T = pntrs$T_fn, R = pntrs$R_fn,
          Z_gn = pntrs$Z_gn, T_gn = pntrs$T_gn,
          theta = c(log_sigma_y = log(sigma_y), 
                    log_sigma_x1 = log(sigma_x[1]), 
                    log_sigma_x2 = log(sigma_x[2]),
                    log_sigma_x3 = log(sigma_x[3]),
                    log_sigma_x3 = log(sigma_x[4])),
          log_prior_pdf = pntrs$log_prior_pdf,
          known_params = known_params, 
          n_states = 4, n_etas = 4)

        infer_result <- ukf(model_nlg, alpha = 0.01, beta = 2, kappa = 1)
            """)
        r_result = ro.r("infer_result")

        observation = np.array(ro.r("y"))
        size_y, observation = check_y(observation.astype("float32"))  # return time length and feature numbers
        num_timesteps, observation_size = size_y

        model_obj = NonlinearSSM.create_model(num_timesteps=num_timesteps,
                                              observation_size=observation_size,
                                              latent_size=4,
                                              initial_state_mean=np.array([0, 1, 0.5, 0]),
                                              initial_state_cov=np.diag([1, 2, 1, 1.5]),
                                              state_noise_std=np.diag([0.1, 0.1, 0.2, 0.1]),
                                              obs_noise_std=np.diag([0.1, 0.1, 0.1]),
                                              dt=0.3,
                                              nonlinear_type="nlg_mv_model")
        infer_result = model_obj.ensemble_Kalman_filter(observation, num_particles=20)
        infer_result_standalone = ensemble_kalman_filter(model_obj, observation, num_particles=20)

        true_state = np.array(ro.r("x"))[..., None]
        plt.plot(infer_result_standalone[0][:, 0].numpy(), color='blue', linewidth=1)
        plt.plot(r_result[0][:, 0], color='green', linewidth=1)
        plt.plot(true_state[:, 0], '-.', color='red', linewidth=1)
        plt.show()
        for i in range(true_state.shape[-1]):
            print('MSE error of the EKF from bssm: %.4f' % np.sum((r_result[1][:,i] - true_state[:,i]) ** 2))
            print('MSE error of the EnKF from TFP: %.4f' % np.sum((infer_result[0].numpy()[:,i] - true_state[:,i]) ** 2))
