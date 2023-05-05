import pytest
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from Models.bssm_model import *
from Inference.KF.kalman_smoother import KalmanSmoother
import os.path as pth
import os


# Automatic convertion between R and Python objects
numpy2ri.activate()

# Must call the function via the package, if the desired function is
# linked with other functions in the package. The simple call via ro.r("""source""")
# will only create a simple object that miss the link.
base = importr('base', lib_loc="/usr/lib/R/library")
bssm = importr('bssm', lib_loc=f"{pth.expanduser('~')}/R/x86_64-pc-linux-gnu-library/4.3")
stat = importr('stats', lib_loc="/usr/lib/R/library")


class TestKalmanSmoother:
    """
    Test Kalman Smoother
    """

################### Test ssm_ulg ##################################

    def test_ksmoother_ssmulg(self):
        # define data
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

        # define ssm
        r_result = ro.r["infer_result"]

        model_obj = SSModel(model_name="ssm_ulg", y=np.array(ro.r["y"]),
                            obs_mtx=np.array(ro.r("Z")), obs_mtx_noise=ro.r("H"),
                            init_theta=ro.r("""c(1, 0.1, 0.1)"""),
                            state_mtx=np.array(ro.r("T")), state_mtx_noise = np.array(ro.r("R")),
                            prior_mean=np.array(ro.r("a1")), prior_cov = np.array(ro.r("P1")),
                          input_state=ro.r("""matrix(0, 3, 1)"""), input_obs=np.array([0.]),
                         )
        model_type_case = model_obj.model_type_case()
        model_obj = model_obj._toRssmulg(prior_fn=ro.r("prior_fn"), update_fn=ro.r("update_fn"))
        infer_result = KalmanSmoother(model_type="linear_gaussian", model=model_obj,
                                      model_type_case=model_type_case).infer_result

        # The R code would start from n+1, so need to remove one column
        g = base.all_equal(r_result[0], infer_result[0][:-1, :])
        assert g[0] == True
        g = base.all_equal(r_result[1], infer_result[1][:, :, :-1])
        assert g[0] == True