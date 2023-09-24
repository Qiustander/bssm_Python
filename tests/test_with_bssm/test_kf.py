import pytest
import tensorflow as tf
import numpy as np
import rpy2.robjects as ro
from Models.check_argument import *
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from Models.ssm_lg import LinearGaussianSSM
import os.path as pth
import os
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


class TestKalmanFilterULG:
    """
    Test Kalman Filter - ssm_ulg
    """
    ro.r("""
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
    infer_result <- kfilter(model_r)
    """)

    @pytest.mark.parametrize(
        ("y", "obs_mtx", "obs_mtx_noise", "init_theta", "state_mtx", "state_mtx_noise",
         "prior_mean", "prior_cov", "input_state", "input_obs", "r_result"),
        [(
         np.array(ro.r["y"]), np.array(ro.r("Z")), np.array(ro.r("H")), ro.r("""c(1, 0.1, 0.1)"""), np.array(ro.r("T")),
         np.array(ro.r("R")), np.array(ro.r("a1")), np.array(ro.r("P1")), ro.r("""matrix(0, 3, 1)"""),
         np.array([0.]), ro.r["infer_result"]
         )])
    def test_kffilter_lg_TFP(self, y, obs_mtx, obs_mtx_noise, init_theta, state_mtx,
                             state_mtx_noise, prior_mean, prior_cov, input_state, input_obs, r_result):
        r_result = r_result
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
        infer_result = model_obj.forward_filter(tf.convert_to_tensor(observation, dtype=model_obj.dtype))
        # compare loglik
        tf.debugging.assert_near(r_result[-1], infer_result.log_likelihoods.numpy().sum(), atol=1e-2)
        # compare filtered_means
        tf.debugging.assert_near(r_result[1], infer_result.filtered_means.numpy(), atol=1e-2)
        # compare filtered_covs
        tf.debugging.assert_near(r_result[3], infer_result.filtered_covs.numpy().transpose(1, 2, 0), atol=1e-2)
        # compare predicted_means
        tf.debugging.assert_near(r_result[0][1:, ...], infer_result.predicted_means.numpy(), atol=1e-2)
        # compare predicted_covs
        tf.debugging.assert_near(r_result[2][..., 1:], infer_result.predicted_covs.numpy().transpose(1, 2, 0),
                                 atol=1e-2)


class TestKalmanFilterMLG:
    """
    Test Kalman Filter - ssm_mlg, local level model
    """
    # define data
    ro.r("""
        data("GlobalTemp", package = "KFAS")
        model_r <- ssm_mlg(GlobalTemp, H = matrix(c(0.15,0.05,0, 0.05), 2, 2),
          R = 0.05, Z = matrix(1, 2, 1), T = 1, P1 = 10,
          state_names = "temperature",
          # using default values, but being explicit for testing purposes
          D = matrix(0, 2, 1), C = matrix(0, 1, 1))
      infer_result <- kfilter(model_r)
    """)

    @pytest.mark.parametrize(
        ("y", "obs_mtx", "obs_mtx_noise", "init_theta", "state_mtx", "state_mtx_noise",
         "prior_mean", "prior_cov", "input_state", "input_obs", "r_result"),
        [(np.array(ro.r("model_r$y")), np.array(ro.r("model_r$Z")), ro.r("model_r$H"), ro.r("model_r$theta"),
          np.array(ro.r("model_r$T")), np.array(ro.r("model_r$R")), np.array(ro.r("model_r$a1")),
          np.array(ro.r("model_r$P1")), np.array(ro.r("model_r$C")), np.array(ro.r("model_r$D")),
          ro.r["infer_result"]
          )])
    def test_kffilter_lg_TFP(self, y, obs_mtx, obs_mtx_noise, init_theta, state_mtx,
                             state_mtx_noise, prior_mean, prior_cov, input_state, input_obs, r_result):
        r_result = r_result
        size_y, observation = check_y(y)
        num_timesteps, observation_size = size_y
        model_obj = LinearGaussianSSM.create_model(num_timesteps=num_timesteps,
                                                   observation_size=observation_size,
                                                   latent_size=1,
                                                   initial_state_mean=prior_mean,
                                                   initial_state_cov=prior_cov,
                                                   state_noise_std=state_mtx_noise,
                                                   obs_noise_std=obs_mtx_noise,
                                                   obs_mtx=obs_mtx,
                                                   state_mtx=state_mtx)

        infer_result = model_obj.forward_filter(tf.convert_to_tensor(observation, dtype=model_obj.dtype))
        # compare loglik
        tf.debugging.assert_near(r_result[-1], infer_result.log_likelihoods.numpy().sum(), atol=1e-2)
        # compare filtered_means
        tf.debugging.assert_near(r_result[1], infer_result.filtered_means.numpy(), atol=1e-2)
        # compare filtered_covs
        tf.debugging.assert_near(r_result[3], infer_result.filtered_covs.numpy().transpose(1, 2, 0), atol=1e-2)
        # compare predicted_means
        tf.debugging.assert_near(r_result[0][1:, ...], infer_result.predicted_means.numpy(), atol=1e-2)
        # compare predicted_covs
        tf.debugging.assert_near(r_result[2][..., 1:], infer_result.predicted_covs.numpy().transpose(1, 2, 0),
                                 atol=1e-2)


class TestKalmanFilterMLG2:
    """
    Test Kalman Filter - ssm_mlg, 3-dim tracking model
    """
    # define data
    ro.r("""
        a1 <- rep(0, 6)
        P1 <- rep (10, 6)
        n <- 100
        sd_x <- c(0.1,0.1,0.02)
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
      infer_result <- kfilter(model_r)
    """)

    @pytest.mark.parametrize(
        ("y", "obs_mtx", "obs_mtx_noise", "init_theta", "state_mtx", "state_mtx_noise",
         "prior_mean", "prior_cov", "input_state", "input_obs", "r_result"),
        [(np.array(ro.r("model_r$y")), np.array(ro.r("model_r$Z")), ro.r("model_r$H"), ro.r("model_r$theta"),
          np.array(ro.r("model_r$T")), np.array(ro.r("model_r$R")), np.array(ro.r("model_r$a1")),
          np.array(ro.r("model_r$P1")), np.array(ro.r("model_r$C")), np.array(ro.r("model_r$D")),
          ro.r["infer_result"]
          )])
    def test_kffilter_lg_TFP(self, y, obs_mtx, obs_mtx_noise, init_theta, state_mtx,
                             state_mtx_noise, prior_mean, prior_cov, input_state, input_obs, r_result):
        r_result = r_result
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

        infer_result = model_obj.forward_filter(tf.convert_to_tensor(observation, dtype=model_obj.dtype))
        # compare loglik
        tf.debugging.assert_near(r_result[-1], infer_result.log_likelihoods.numpy().sum(), atol=1e-2)
        # compare filtered_means
        tf.debugging.assert_near(r_result[1], infer_result.filtered_means.numpy(), atol=1e-2)
        # compare filtered_covs
        tf.debugging.assert_near(r_result[3], infer_result.filtered_covs.numpy().transpose(1, 2, 0), atol=1e-2)
        # compare predicted_means
        tf.debugging.assert_near(r_result[0][1:, ...], infer_result.predicted_means.numpy(), atol=1e-2)
        # compare predicted_covs
        tf.debugging.assert_near(r_result[2][..., 1:], infer_result.predicted_covs.numpy().transpose(1, 2, 0),
                                 atol=1e-2)
