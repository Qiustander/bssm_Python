import pytest
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from Models.check_argument import *
from numpy.testing import assert_array_almost_equal_nulp
import os.path as pth
import os

# Automatic convertion between R and Python objects
numpy2ri.activate()

base = importr('base', lib_loc="/usr/lib/R/library")

# Define R objects as a multiline string
r_code = """
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
a1 <- rep(0.2, 3)
P1 <- diag(10, 3)
"""



# Execute R code
ro.r(r_code)

# Convert R objects to Python
n = ro.r("n")[0]
y = np.array(ro.r("y"))

obs_mtx = np.array(ro.r("Z"))
noise_std = ro.r("H")[0]
state_mtx = np.array(ro.r("T"))
state_mtx_lower = np.array(ro.r("R"))
prior_mean = np.array(ro.r("a1"))
prior_cov = np.array(ro.r("P1"))

# Import R function
ro.r("""source('{path_name}')""".
     format(path_name=pth.join(pth.abspath(pth.join(os.getcwd(), os.pardir, os.pardir)), 'bssm_R/R/check_arguments.R')))
r_check_y = ro.globalenv['check_y']
r_check_obsmtx = ro.globalenv['check_Z']
r_check_statemtx = ro.globalenv['check_T']
r_check_statemtxlow = ro.globalenv['check_R']
r_check_priormean = ro.globalenv['check_a1']
r_check_priorvar = ro.globalenv['check_P1']
r_check_inputobs = ro.globalenv['check_D']
r_check_inputstate = ro.globalenv['check_C']
r_check_noisestd = ro.globalenv['check_H']


@pytest.mark.parametrize(
    ("y"),
    [y
    ],
)
class TestY:
    """
    Test y - observation
    """

    def test_uni_y(self, y):
        size_y = check_y(y, multivariate=False, distribution="gaussian")
        n= size_y[0]
        yr = r_check_y(ro.FloatVector(y), multivariate=False, distribution="gaussian")
        nr = yr.shape[0]
        assert_array_almost_equal_nulp(n, nr)

    def test_na_y(self, y):
        y[np.random.choice(y.size, y.size//8)] = np.nan
        size_y = check_y(y, multivariate=False, distribution="gaussian")
        n= size_y[0]
        yr = r_check_y(ro.FloatVector(y), multivariate=False, distribution="gaussian")
        nr = yr.shape[0]
        assert_array_almost_equal_nulp(n, nr)

    def test_mul_y(self, y):
        y_mul = np.repeat(y[..., None], 5, axis=-1)
        size_y = check_y(y_mul, multivariate=True, distribution="gaussian")
        n, p = size_y
        yr = r_check_y(y_mul, multivariate=True, distribution="gaussian")
        nr, pr = yr.shape[0], yr.shape[1]
        assert_array_almost_equal_nulp(n, nr)

    def test_inf_y(self, y):
        y[np.random.choice(y.size, y.size//8)] = np.inf
        with pytest.raises(TypeError):
            check_y(y, multivariate=False, distribution="gaussian")

    def test_dist_y(self, y):
        ysize = y.size
        #"poisson"
        y = np.random.poisson(2, ysize) + 0.1
        with pytest.raises(AssertionError):
            check_y(y, multivariate=False, distribution="poisson")
        # "negative binomial"
        y = np.random.negative_binomial(ysize//3, 0.3, size=ysize) + 0.1
        with pytest.raises(AssertionError):
            check_y(y, multivariate=False, distribution="negative binomial")
        #  "binomial"
        y = np.random.binomial(ysize//3, 0.3, size=ysize) + 0.1
        with pytest.raises(AssertionError):
            check_y(y, multivariate=False, distribution="binomial")
        y[np.random.choice(y.size, y.size//8)] = -0.5
        with pytest.raises(AttributeError):
            check_y(y, multivariate=False, distribution="binomial")




@pytest.mark.parametrize(
    ("obs_mtx"),
    [obs_mtx
    ],
)
class TestObsMtx:
    """
    Test Z - observation matrix. Either a vector of length m, a m x n matrix.
    """

    def test_uni_Z(self, obs_mtx):
        obs_mtx_Py = check_obs_mtx(obs_mtx, p=1, n=y.size, multivariate=False)
        obs_mtx_r = r_check_obsmtx(obs_mtx, p=1, n=y.size, multivariate=False)
        assert_array_almost_equal_nulp(obs_mtx_Py, obs_mtx_r)

    def test_vec_uni_Z(self, obs_mtx):
        obs_mtx_Py = check_obs_mtx(obs_mtx[:, 0:1], p=1, n=y.size, multivariate=False)
        obs_mtx_r = r_check_obsmtx(obs_mtx[:, 0:1], p=1, n=y.size, multivariate=False)
        assert_array_almost_equal_nulp(obs_mtx_Py, obs_mtx_r)

    def test_mul_Z(self, obs_mtx):
        obs_mtx = np.repeat(obs_mtx[None, ...], 3, axis=0)
        obs_mtx_Py = check_obs_mtx(obs_mtx, p=3, n=y.size, multivariate=True)
        obs_mtx_r = r_check_obsmtx(obs_mtx, p=3, n=y.size, multivariate=True)
        assert_array_almost_equal_nulp(obs_mtx_Py, obs_mtx_r)


@pytest.mark.parametrize(
    ("state_mtx_lower"),
    [state_mtx_lower
    ],
)
class TestStateMtxLower:
    """
    Test R - lower state matrix. Either a m x k matrix or a m x k x n array.
    """

    def test_mtx_R(self, state_mtx_lower):
        state_mtx_Py = check_mtx_lower(state_mtx_lower, m=state_mtx_lower.shape[0], n=y.size)
        state_mtx_r = r_check_statemtxlow(state_mtx_lower, m=state_mtx_lower.shape[0], n=y.size)
        assert_array_almost_equal_nulp(state_mtx_Py, state_mtx_r)
        assert state_mtx_lower.shape[0] == state_mtx_lower.shape[1]

    def test_array_R(self, state_mtx_lower):
        state_mtx_lower = np.repeat(state_mtx_lower[..., None], n, axis=-1)
        state_mtx_Py = check_mtx_lower(state_mtx_lower, m=state_mtx_lower.shape[0], n=y.size)
        state_mtx_r = r_check_statemtxlow(state_mtx_lower, m=state_mtx_lower.shape[0], n=y.size)
        assert_array_almost_equal_nulp(state_mtx_Py, state_mtx_r)
        assert state_mtx_lower.shape[0] == state_mtx_lower.shape[1]

    def test_scalar_R(self, state_mtx_lower):
        state_mtx_lower = np.array([1])
        state_mtx_Py = check_mtx_lower(state_mtx_lower, m=1, n=y.size)
        state_mtx_r = r_check_statemtxlow(state_mtx_lower, m=1, n=y.size)
        assert_array_almost_equal_nulp(state_mtx_Py, state_mtx_r)
        assert state_mtx_lower.shape[0] == 1


@pytest.mark.parametrize(
    ("state_mtx"),
    [state_mtx
    ],
)
class TestStateMtx:
    """
    Test T - state matrix. Either a m x m matrix or a m x m x n array.
    """

    def test_mtx_T(self, state_mtx):
        state_mtx_Py = check_state_mtx(state_mtx, m=state_mtx.shape[0], n=y.size)
        state_mtx_r = r_check_statemtx(state_mtx, m=state_mtx.shape[0], n=y.size)
        assert_array_almost_equal_nulp(state_mtx_Py, state_mtx_r)
        assert state_mtx.shape[0] == state_mtx.shape[1]

    def test_array_T(self, state_mtx):
        state_mtx = np.repeat(state_mtx[..., None], n, axis=-1)
        state_mtx_Py = check_state_mtx(state_mtx, m=state_mtx.shape[0], n=y.size)
        state_mtx_r = r_check_statemtx(state_mtx, m=state_mtx.shape[0], n=y.size)
        assert_array_almost_equal_nulp(state_mtx_Py, state_mtx_r)
        assert state_mtx.shape[0] == state_mtx.shape[1]

    def test_scalar_T(self, state_mtx):
        state_mtx = np.array([1])
        state_mtx_Py = check_state_mtx(state_mtx, m=1, n=y.size)
        state_mtx_r = r_check_statemtx(state_mtx, m=1, n=y.size)
        assert_array_almost_equal_nulp(state_mtx_Py, state_mtx_r)
        assert state_mtx.shape[0] == 1


@pytest.mark.parametrize(
    ("prior_mean"),
    [prior_mean
    ],
)
class TestPriorMean:
    """
    Test a1 - prior mean for the initial state as a vector of length m.
    """

    def test_m_mean(self, prior_mean):
        prior_mean_Py = check_prior_mean(prior_mean, m=state_mtx.shape[0])
        prior_mean_r = r_check_priormean(prior_mean, m=state_mtx.shape[0])
        assert_array_almost_equal_nulp(prior_mean_Py, prior_mean_r)


    def test_scalar_mean(self, prior_mean):
        prior_mean_Py = check_prior_mean(prior_mean, m=state_mtx.shape[0])
        prior_mean_r = r_check_priormean(prior_mean, m=state_mtx.shape[0])
        assert_array_almost_equal_nulp(prior_mean_Py, prior_mean_r)
        assert state_mtx.shape[0] == state_mtx.shape[0]

    def test_None_mean(self, prior_mean):
        prior_mean_Py = check_prior_mean(None, m=state_mtx.shape[0])
        prior_mean_r = r_check_priormean(ro.NULL, m=state_mtx.shape[0])
        assert_array_almost_equal_nulp(prior_mean_Py, prior_mean_r)
        assert state_mtx.shape[0] == state_mtx.shape[0]


@pytest.mark.parametrize(
    ("prior_cov"),
    [prior_cov
    ],
)
class TestPriorCov:
    """
    Test P1 - prior covariance for the initial state as m x m matrix.
    """

    def test_m_cov(self, prior_cov):
        prior_cov_Py = check_prior_cov(prior_cov, m=state_mtx.shape[0])
        prior_cov_r = r_check_priorvar(prior_cov, m=state_mtx.shape[0])
        assert_array_almost_equal_nulp(prior_cov_Py, prior_cov_r)


    def test_scalar_cov(self, prior_cov):
        prior_cov_Py = check_prior_cov(prior_cov, m=state_mtx.shape[0])
        prior_cov_r = r_check_priorvar(prior_cov, m=state_mtx.shape[0])
        assert_array_almost_equal_nulp(prior_cov_Py, prior_cov_r)
        assert state_mtx.shape[0] == state_mtx.shape[0]

    def test_None_cov(self, prior_cov):
        prior_cov_Py = check_prior_cov(None, m=state_mtx.shape[0])
        prior_cov_r = r_check_priorvar(ro.NULL, m=state_mtx.shape[0])
        assert_array_almost_equal_nulp(prior_cov_Py, prior_cov_r)
        assert state_mtx.shape[0] == state_mtx.shape[0]


class TestInputObs:
    """
    Test Dt - intercept terms \eqn{D_t} for the observations equation,
    given as a scalar or vector of length n.
    """

    def test_n_input(self):
        p = 5
        input_ob = np.random.rand(p, y.size)
        input_obs_Py = check_input_obs(input_ob, p=p, n=y.size)
        input_obs_r = r_check_inputobs(input_ob, p=p, n=y.size)
        assert_array_almost_equal_nulp(input_obs_Py, input_obs_r)

        input_ob = np.random.rand(p, 1)
        input_obs_Py = check_input_obs(input_ob, p=p, n=1)
        input_obs_r = r_check_inputobs(input_ob, p=p, n=1)
        assert_array_almost_equal_nulp(input_obs_Py, input_obs_r)

    def test_scalar_input(self):
        p = 1
        input_ob = np.random.rand(1)
        input_obs_Py = check_input_obs(input_ob,  p=1, n=y.size)
        input_obs_r = r_check_inputobs(input_ob, p=1, n=y.size)
        assert_array_almost_equal_nulp(input_obs_Py, input_obs_r)

    def test_None_input(self):
        input_obs_Py = check_input_obs(None, p=1, n=y.size)
        input_obs_r = r_check_inputobs(ro.NULL, p=1, n=y.size)
        assert_array_almost_equal_nulp(input_obs_Py, input_obs_r)



@pytest.mark.parametrize(
    ("m"),
    [5
    ],
)
class TestInputState:
    """
    Test Ct - Intercept terms \eqn{C_t} for the state equation,
    given as a m x 1 or m x n matrix.
    """

    def test_n_input(self, m):
        input_ob = np.random.rand(m, y.size)
        input_state_Py = check_input_state(input_ob, m=m, n=y.size)
        input_state_r = r_check_inputstate(input_ob, m=m, n=y.size)
        assert_array_almost_equal_nulp(input_state_Py, input_state_r)

        input_ob = np.random.rand(m, 1)
        input_state_Py = check_input_state(input_ob, m=m, n=1)
        input_state_r = r_check_inputstate(input_ob, m=m, n=1)
        assert_array_almost_equal_nulp(input_state_Py, input_state_r)

    def test_None_input(self, m):
        input_state_Py = check_input_state(None, m=m, n=y.size)
        input_state_r = r_check_inputstate(ro.NULL, m=m, n=y.size)
        assert_array_almost_equal_nulp(input_state_Py, input_state_r)


@pytest.mark.parametrize(
    "n, p",
    [(y.size, 6)
    ],
)
class TestNoiseSTD:
    """
    Test H - Check vector H of standard deviations of noise. Either a scalar or a vector of  length n,
        or p x p matrix or p x p x n array.
    """

    def test_vector_std(self, n, p):
        input_ob = np.random.rand(n)
        input_obs_Py = check_noise_std(input_ob, p=1, n=n, multivariate=False)
        input_obs_r = r_check_noisestd(input_ob, p=1, n=n, multivariate=False)
        assert_array_almost_equal_nulp(input_obs_Py, input_obs_r)

    def test_scalar_std(self, n, p):
        input_ob = np.random.rand((1))
        input_obs_Py = check_noise_std(input_ob,  p=1, n=n, multivariate=False)
        input_obs_r = r_check_noisestd(input_ob, p=1, n=n, multivariate=False)
        assert_array_almost_equal_nulp(input_obs_Py, input_obs_r)

    def test_matrix_std(self, n, p):
        input_ob = np.random.rand(p, p)
        input_obs_Py = check_noise_std(input_ob,  p=p, n=y.size, multivariate=True)
        input_obs_r = r_check_noisestd(input_ob, p=p, n=y.size, multivariate=True)
        assert_array_almost_equal_nulp(input_obs_Py, input_obs_r)

    def test_array_std(self, n, p):
        input_ob = np.random.rand(p, p, n)
        input_obs_Py = check_noise_std(input_ob,  p=p, n=y.size, multivariate=True)
        input_obs_r = r_check_noisestd(input_ob, p=p, n=y.size, multivariate=True)
        assert_array_almost_equal_nulp(input_obs_Py, input_obs_r)


