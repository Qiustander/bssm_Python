import pytest
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from Models.bssm_model import *
from numpy.testing import assert_array_almost_equal_nulp
import os.path as pth
import os


# Automatic convertion between R and Python objects
numpy2ri.activate()

base = importr('base', lib_loc="/usr/lib/R/library")
bssm_package = importr("callr", lib_loc="/home/project/R/x86_64-pc-linux-gnu-library/4.2")

# Import R function
ro.r("""source('{path_name}')""".
     format(path_name=pth.join(pth.abspath(pth.join(os.getcwd(), os.pardir, os.pardir)), 'bssm_R/R/models.R')))


class TestModel:
    """
    Test bssm Model
    """

################### Test ssm_ulg ##################################
    def test_ssmulg(self):
        pass

    def test_ssmulg_with_r(self):
        r_ssm_def = ("""
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
        
        model <- ssm_ulg(y, Z, H, T, R, a1, P1,
          init_theta = c(1, 0.1, 0.1),
          update_fn = update_fn, prior_fn = prior_fn,
          state_names = c("level", "b1", "b2"),
          # using default values, but being explicit for testing purposes
          C = matrix(0, 3, 1), D = numeric(1))
         }""")
        r_ssm = ro.r("model")
        model_obj = SSModel(model_name="ssm_ulg", y=np.array(ro.r["y"][0]),
                            obs_mtx=np.array(ro.r("Z")), noise_std=ro.r("H")[0],
                            state_mtx=np.array(ro.r("T")),state_mtx_lower = np.array(ro.r("R")),
                            prior_mean=np.array(ro.r("a1")), prior_cov = np.array(ro.r("P1")),
                          input_state=np.array(ro.r("C")), input_obs=np.array(ro.r("D")),
                         )._toR()
        comp_result = base.all_equal(r_ssm, model_obj)
        assert comp_result[0] == True
