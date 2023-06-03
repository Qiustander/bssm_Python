import pytest
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
import matplotlib.pyplot as plt
from Models.bssm_model import *
from Inference.Kalman.kalman_smoother import KalmanSmoother as ksmoother
from Models.ssm_lg import LinearGaussianSSM
import os.path as pth
import os

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


class TestKalmanSmootherULG:
    """
    Test Kalman Smoother - ssm_ulg
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
    infer_result <- smoother(model_r)
    """)

    @pytest.mark.parametrize(
        ("y", "obs_mtx", "obs_mtx_noise", "init_theta", "state_mtx", "state_mtx_noise",
         "prior_mean", "prior_cov", "input_state", "input_obs", "r_result"),
        [(
         np.array(ro.r["y"]), np.array(ro.r("Z")), np.array(ro.r("H")), ro.r("""c(1, 0.1, 0.1)"""), np.array(ro.r("T")),
         np.array(ro.r("R")), np.array(ro.r("a1")), np.array(ro.r("P1")), ro.r("""matrix(0, 3, 1)"""),
         np.array([0.]), ro.r["infer_result"]
         )])
    def test_ksmoother_standalone_TFP(self, y, obs_mtx, obs_mtx_noise, init_theta, state_mtx,
                                      state_mtx_noise, prior_mean, prior_cov, input_state, input_obs, r_result):
        r_result = r_result

        model_obj = SSModel(model_name="ssm_ulg", y=y, state_dim=state_mtx.shape[0],
                            obs_mtx=obs_mtx, obs_mtx_noise=obs_mtx_noise,
                            init_theta=init_theta,
                            state_mtx=state_mtx, state_mtx_noise=state_mtx_noise,
                            prior_mean=prior_mean, prior_cov=prior_cov,
                            input_state=input_state, input_obs=input_obs,
                            )
        infer_result = ksmoother(model_type="linear_gaussian", model=model_obj).infer_result
        # compare smoothed_means
        tf.debugging.assert_near(r_result[0], infer_result[0].numpy(), atol=1e-4)
        # compare smoothed_covs
        tf.debugging.assert_near(r_result[1], infer_result[1].numpy().transpose(1, 2, 0), atol=1e-4)

    @pytest.mark.parametrize(
        ("y", "obs_mtx", "obs_mtx_noise", "init_theta", "state_mtx", "state_mtx_noise",
         "prior_mean", "prior_cov", "input_state", "input_obs", "r_result"),
        [(
         np.array(ro.r["y"]), np.array(ro.r("Z")), np.array(ro.r("H")), ro.r("""c(1, 0.1, 0.1)"""), np.array(ro.r("T")),
         np.array(ro.r("R")), np.array(ro.r("a1")), np.array(ro.r("P1")), ro.r("""matrix(0, 3, 1)"""),
         np.array([0.]), ro.r["infer_result"]
         )])
    def test_ksmoother_lg_TFP(self, y, obs_mtx, obs_mtx_noise, init_theta, state_mtx,
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
        infer_result = model_obj.posterior_marginals(tf.convert_to_tensor(observation, dtype=model_obj.dtype))
        # compare smoothed_means
        tf.debugging.assert_near(r_result[0], infer_result[0].numpy(), atol=1e-4)
        # compare smoothed_covs
        tf.debugging.assert_near(r_result[1], infer_result[1].numpy().transpose(1, 2, 0), atol=1e-4)


class TestKalmanSmootherMLG:
    """
    Test Kalman Smoother - ssm_mlg
    """
    # define data
    ro.r("""
        data("GlobalTemp", package = "KFAS")
        model_r <- ssm_mlg(GlobalTemp, H = matrix(c(0.15,0.05,0, 0.05), 2, 2),
          R = 0.05, Z = matrix(1, 2, 1), T = 1, P1 = 10,
          state_names = "temperature",
          # using default values, but being explicit for testing purposes
          D = matrix(0, 2, 1), C = matrix(0, 1, 1))
      infer_result <- smoother(model_r)
    """)

    @pytest.mark.parametrize(
        ("y", "obs_mtx", "obs_mtx_noise", "init_theta", "state_mtx", "state_mtx_noise",
         "prior_mean", "prior_cov", "input_state", "input_obs", "r_result"),
        [(np.array(ro.r("model_r$y")), np.array(ro.r("model_r$Z")), ro.r("model_r$H"), ro.r("model_r$theta"),
          np.array(ro.r("model_r$T")), np.array(ro.r("model_r$R")), np.array(ro.r("model_r$a1")),
          np.array(ro.r("model_r$P1")), np.array(ro.r("model_r$C")), np.array(ro.r("model_r$D")),
          ro.r["infer_result"]
          )])
    def test_ksmoother_standalone_TFP(self, y, obs_mtx, obs_mtx_noise, init_theta, state_mtx,
                                      state_mtx_noise, prior_mean, prior_cov, input_state, input_obs, r_result):
        r_result = r_result

        model_obj = SSModel(model_name="ssm_ulg", y=y, state_dim=state_mtx.shape[0],
                            obs_mtx=obs_mtx, obs_mtx_noise=obs_mtx_noise,
                            init_theta=init_theta,
                            state_mtx=state_mtx, state_mtx_noise=state_mtx_noise.squeeze(),
                            prior_mean=prior_mean, prior_cov=prior_cov,
                            input_state=input_state, input_obs=input_obs,
                            )
        infer_result = ksmoother(model_type="linear_gaussian", model=model_obj).infer_result
        # compare smoothed_means
        tf.debugging.assert_near(r_result[0], infer_result[0].numpy(), atol=1e-4)
        # compare smoothed_covs
        tf.debugging.assert_near(r_result[1], infer_result[1].numpy().transpose(1, 2, 0), atol=1e-4)

    @pytest.mark.parametrize(
        ("y", "obs_mtx", "obs_mtx_noise", "init_theta", "state_mtx", "state_mtx_noise",
         "prior_mean", "prior_cov", "input_state", "input_obs", "r_result"),
        [(np.array(ro.r("model_r$y")), np.array(ro.r("model_r$Z")), ro.r("model_r$H"), ro.r("model_r$theta"),
          np.array(ro.r("model_r$T")), np.array(ro.r("model_r$R")), np.array(ro.r("model_r$a1")),
          np.array(ro.r("model_r$P1")), np.array(ro.r("model_r$C")), np.array(ro.r("model_r$D")),
          ro.r["infer_result"]
          )])
    def test_ksmoother_lg_TFP(self, y, obs_mtx, obs_mtx_noise, init_theta, state_mtx,
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
        infer_result = model_obj.posterior_marginals(tf.convert_to_tensor(observation, dtype=model_obj.dtype))
        # compare smoothed_means
        tf.debugging.assert_near(r_result[0], infer_result[0].numpy(), atol=1e-4)
        # compare smoothed_covs
        tf.debugging.assert_near(r_result[1], infer_result[1].numpy().transpose(1, 2, 0), atol=1e-4)


class TestKalmanSmootherMLG2:
    """
    Test Kalman Smoother - ssm_mlg
    """
    # define data
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
      infer_result <- smoother(model_r)
    """)

    @pytest.mark.parametrize(
        ("y", "obs_mtx", "obs_mtx_noise", "init_theta", "state_mtx", "state_mtx_noise",
         "prior_mean", "prior_cov", "input_state", "input_obs", "r_result"),
        [(np.array(ro.r("model_r$y")), np.array(ro.r("model_r$Z")), ro.r("model_r$H"), ro.r("model_r$theta"),
          np.array(ro.r("model_r$T")), np.array(ro.r("model_r$R")), np.array(ro.r("model_r$a1")),
          np.array(ro.r("model_r$P1")), np.array(ro.r("model_r$C")), np.array(ro.r("model_r$D")),
          ro.r["infer_result"]
          )])
    def test_ksmoother_standalone_TFP(self, y, obs_mtx, obs_mtx_noise, init_theta, state_mtx,
                                      state_mtx_noise, prior_mean, prior_cov, input_state, input_obs, r_result):
        r_result = r_result

        model_obj = SSModel(model_name="ssm_mlg", y=y, state_dim=state_mtx.shape[0],
                            obs_mtx=obs_mtx, obs_mtx_noise=obs_mtx_noise,
                            init_theta=init_theta,
                            state_mtx=state_mtx, state_mtx_noise=state_mtx_noise.squeeze(),
                            prior_mean=prior_mean, prior_cov=prior_cov,
                            input_state=input_state, input_obs=input_obs,
                            )
        infer_result = ksmoother(model_type="linear_gaussian", model=model_obj).infer_result
        # compare smoothed_means
        tf.debugging.assert_near(r_result[0], infer_result[0].numpy(), atol=1e-3)
        # compare smoothed_covs
        tf.debugging.assert_near(r_result[1], infer_result[1].numpy().transpose(1, 2, 0), atol=1e-4)

    @pytest.mark.parametrize(
        ("y", "state", "obs_mtx", "obs_mtx_noise", "init_theta", "state_mtx", "state_mtx_noise",
         "prior_mean", "prior_cov", "input_state", "input_obs", "r_result"),
        [(np.array(ro.r("model_r$y")), np.array(ro.r("x")), np.array(ro.r("model_r$Z")), ro.r("model_r$H"),
          ro.r("model_r$theta"),
          np.array(ro.r("model_r$T")), np.array(ro.r("model_r$R")), np.array(ro.r("model_r$a1")),
          np.array(ro.r("model_r$P1")), np.array(ro.r("model_r$C")), np.array(ro.r("model_r$D")),
          ro.r["infer_result"]
          )])
    def test_ksmoother_lg_TFP(self, y, state, obs_mtx, obs_mtx_noise, init_theta, state_mtx,
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
        infer_result = model_obj.posterior_marginals(tf.convert_to_tensor(observation, dtype=model_obj.dtype))
        infer_result_kf = model_obj.forward_filter(tf.convert_to_tensor(observation, dtype=model_obj.dtype))

        # compare smoothed_means
        debug_plot(infer_result[0].numpy(), r_result[0], state)
        tf.debugging.assert_near(r_result[0], infer_result[0].numpy(), atol=1e-2)
        # compare smoothed_covs
        tf.debugging.assert_near(r_result[1], infer_result[1].numpy().transpose(1, 2, 0), atol=1e-4)


class TestKalmanSmootherAR:
    """
    Test Kalman Smoother - ssm_ar1_lg
    """
    # define data
    ro.r("""
        n <- 200
        mu <- 2
        rho <- 0.7
        sd_y <- 0.1
        sigma <- 0.5
        beta <- -1
        x <- rnorm(n)
        z <- y <- numeric(n)
        z[1] <- rnorm(1, mu, sigma / sqrt(1 - rho^2))
        y[1] <- rnorm(1, beta * x[1] + z[1], sd_y)
        for(i in 2:n) {
          z[i] <- rnorm(1, mu * (1 - rho) + rho * z[i - 1], sigma)
          y[i] <- rnorm(1, beta * x[i] + z[i], sd_y)
        }
        model_r <- ar1_lg(y, rho = uniform(0.5, -1, 1),
          sigma = halfnormal(1, 10), mu = normal(0, 0, 1),
          sd_y = halfnormal(1, 10),
          xreg = x,  beta = normal(0, 0, 1))
          infer_result <- smoother(model_r)
            """)

    @pytest.mark.parametrize(
        ("y", "obs_mtx_noise", "rho", "mu", "state_mtx_noise", "xreg", "beta", "r_result"),
        [(np.array(ro.r("model_r$y")), ro.r("""halfnormal(1, 10)$init"""), ro.r("""uniform(0.5, -1, 1)$init"""),
          ro.r("""normal(0, 0, 1)$init"""), ro.r("""halfnormal(1, 10)$init"""),
          ro.r["x"], ro.r("""normal(0, 0, 1)"""), ro.r["infer_result"]
          )])
    def test_ksmoother_TFP(self, y, obs_mtx_noise, rho, mu, state_mtx_noise, xreg, beta, r_result):
        r_result = r_result
        model_obj = SSModel(model_name="ar1_lg", y=y, state_dim=1,
                            rho_state=rho, mu_state=mu,
                            obs_mtx_noise=obs_mtx_noise,
                            state_mtx_noise=state_mtx_noise
                            )
        infer_result = ksmoother(model_type="linear_gaussian", model=model_obj).infer_result
        # compare smoothed_means
        tf.debugging.assert_near(r_result[0], infer_result[0].numpy(), atol=1e-4)
        # compare smoothed_covs
        tf.debugging.assert_near(r_result[1], infer_result[1].numpy().transpose(1, 2, 0), atol=1e-4)


def debug_plot(tfp_result, r_result, true_state):
    tfp, = plt.plot(tfp_result, color='blue', linewidth=1)
    r, = plt.plot(r_result, color='green', linewidth=1)
    true, = plt.plot(true_state, '-.', color='red', linewidth=1)
    plt.legend(handles=[tfp, r, true], labels=['smoother', 'filter', 'true'])

    plt.show()
    print(f'Max error of R and TFP: {np.max(np.abs(tfp_result - r_result))}')
