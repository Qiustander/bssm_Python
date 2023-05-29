import numpy as np
from .prior import is_prior, is_prior_list


# TODO: Be Pythonic.. Not expand dim
def check_y(x, multivariate=False, distribution="gaussian"):
    """
    Args:
        x: input time series with length n  (second last dimension), can contain missing values for imputation
        multivariate: whether the time series is multivariate
        distribution: distribution in the multivariate time series

    Returns:
        n (int): length of the time series
    """
    if not np.isnan(x).all():
        if multivariate:
            if not isinstance(x, np.ndarray) or x.ndim != 2:
                raise ValueError("Argument 'y' must be a Tensor or Numpy array")
            if x.shape[0] < 2:
                raise ValueError("Number of rows in 'y', i.e. number of time points, must be > 1.")
        else:
            if not x.shape:
                raise ValueError("Scalar is not accepted.")
            if len(x.shape) == 1:
                x = x[..., None]
            if x.shape[0] < 2:
                raise ValueError("Length of argument y, i.e. number of time points, must be > 1.")
            if distribution != "gaussian" and (np.logical_not(np.isnan(x)) & (x < 0)).any():
                raise AttributeError(f"Negative values not allowed for {distribution} distribution.")
            elif distribution in ["negative binomial", "binomial", "poisson"]:
                finite_x = x[np.isfinite(x)]
                if np.any((finite_x != finite_x.astype(int))):
                    raise AssertionError(f"Non-integer values not allowed for {distribution} distribution.")
        if np.isinf(x).any():
            raise TypeError("Argument 'y' must contain only finite or NA values.")

    return x.shape, x


# For basic time series structure

def check_period(x, n):
    if not isinstance(x, int):
        raise ValueError("Argument 'period' should be a single integer.")
    else:
        if x < 3:
            raise ValueError("Argument 'period' should be an integer larger than 2.")
        if x >= n:
            raise ValueError("Period should be less than the number of time points.")


def check_distribution(x, distribution):
    """Checks that observations are compatible with their distributions are made.
    Args:
        x (np.array): input time serie
        distribution (List[str]): list of distributions

    """
    for i in range(x.shape[1]):
        if distribution[i] != "gaussian" and (np.logical_not(np.isnan(x[:, i])) & (x[:, i] < 0)).any():
            raise ValueError(f"Negative values not allowed for {distribution[i]} distribution.")
        else:
            if distribution[i] in ["negative binomial", "binomial", "poisson"]:
                finite_x = x[:, i][np.isfinite(x[:, i])]
                if np.any((finite_x != finite_x.astype(int))):
                    raise ValueError(f"Non-integer values not allowed for {distribution[i]} distribution.")


def check_sd(x, type, add_prefix=True):
    if add_prefix:
        param = f"sd_{type}"
    else:
        param = type

    if not type(x) in [np.ndarray, float, int]:
        raise ValueError(f"Argument {param} must be numeric.")
    if x < 0:
        raise ValueError(f"Argument {param} must be non-negative.")
    if np.isinf(x):
        raise ValueError(f"Argument {param} must be finite.")


def check_mu(x):
    if isinstance(x, np.ndarray) and len(x) == 1:
        return x
    if type(x) in [float, int, list]:
        return np.array(x)[None]
    if not x.shape:
        x = x[None]
    if len(x) != 1:
        raise ValueError("Argument 'mu' must be of length one.")
    if not np.isfinite(x).all():
        raise ValueError("Argument 'mu' must contain only finite values.")
    raise ValueError("No return.")

def check_rho(x):
        #TODO: unittest to check [x]
        if isinstance(x, np.ndarray) and len(x) == 1:
            return x
        if type(x) in [float, int, list]:
            return np.array(x)[None]
        if not x.shape:
            return x[None]
        if len(x) != 1:
            raise ValueError("Argument 'rho' must be of length one.")
        if abs(x) >= 1:
            raise ValueError("Argument 'rho' must be strictly between -1 and 1.")
        raise ValueError("No return.")

def check_phi(x):
    if x < 0:
        raise ValueError("Parameter 'phi' must be non-negative.")


def check_prior(x, name):
    if not is_prior(x) or is_prior_list(x):
        raise ValueError(f"{name} must belong 'bssm_prior' or 'bssm_prior_list'.")


def check_prop(x):
    if not type(x) in [np.ndarray, float, int] or x >= 1 or x <= 0:
        raise ValueError("Argument must be on interval (0, 1).")


##### Check input C & D

def check_input_obs(x, p, n):
    """Check intercept terms \eqn{D_t} for the observations equation,
        given as a scalar, vector n, vector p, or matrix p x n.
    Args:
        x:
        p:
        n:

    Returns:

    """
    #TODO unit test for p <1
    # TODO unittest raise error when input is float
    if x is None:
        x = np.zeros([p,])
    else:
        if type(x) in [float, int, list]:
            x = np.array(x)
        elif not isinstance(x, np.ndarray):
            raise ValueError("input obs must be numeric.")
        x = x.squeeze()
        if p == 1: #univariate
            if not x.shape:
                x = x[None]
            if not np.size(x) in (1, n):
                raise ValueError("input obs variable must be a scalar or length n,"
                                 " where n is the number of observations.")
        else: #multivariate
            if len(x.shape) == 1 and not x.shape[0] == p:
                raise ValueError("input obs variable must be p or p x n matrix, "
                                 "where p is the number of series.")
            elif len(x.shape) == 2 and not x.shape[-1] == n:
                raise ValueError("input obs variable must be p or p x n matrix, "
                                 "where p is the number of series.")
    return x


def check_input_state(x, m, n):
    """Check Intercept terms \eqn{C_t} for the state equation,
    given as a m vector, or m x n matrix.
    Args:
        x:
        m:
        n:

    Returns:

    """
    if x is None:
        x = np.zeros([m,])
    else:
        if type(x) in [float, int, list]:
            x = np.array(x)
        elif not isinstance(x, np.ndarray):
            raise ValueError("input state must be numeric.")
        x = x.squeeze()
        if not x.shape:
            x = x[None]
        elif len(x.shape) == 1 and not x.shape[0] == m:
                raise ValueError("input state variable must be m or m x n matrix, "
                                 "where m is the number of states.")
        elif len(x.shape) == 2 and not x.shape[-1] == n:
            raise ValueError("input state variable must be m or m x n matrix, "
                             "where m is the number of states.")
    return x


def check_obs_mtx_noise(x, p, n):
    """Check Noise coefficient matrix H (lower triangular) for the observed euqation.
        Either a p x p matrix or p x p x n array.
    Args:
        x:
        p:
        n:

    Returns:

    """
    if type(x) in [float, int, list]:
        x = np.array(x)
    elif not isinstance(x, np.ndarray):
        raise ValueError("obs noise must be numeric.")
    x = x.squeeze()
    if p == 1:
        if not x.shape:
            return x[None, None]
        if (x.size > 1 and x.size != n):
            raise ValueError("obs noise covariance must be a scalar or length n, "
                             "where n is the length of the time series y")
    else:
        if len(x.shape) == 2:
            if x.shape != (p, p):
                raise ValueError(
                    "obs noise covariance' must be a p x p matrix "
                    "where p is the number of series and n is the length of the series.")
        elif len(x.shape) == 3:
            if x.shape[:2] != (p, p) and x.shape[2] != n:
                raise ValueError(
                    "obs noise covariance must p x p x n matrix, "
                    "where p is the number of series and n is the length of the series.")

    return x


def check_state_noise(x, m, n):
    """
    Check noise  matrix R (lower triangular) the state equation. Either a m x k matrix or a m x k x n array.
    Args:
        x:
        m:
        n:

    Returns:

    """
    if type(x) in [float, int, list]:
        x = np.array(x)
    elif not isinstance(x, np.ndarray):
        raise ValueError("state noise must be numeric.")
    x = x.squeeze()
    try:
        if not x.shape:
            return x[None, None]
        if x.size==1 and m !=1 :
            raise ValueError("state noise covariance is a scalar now, it must be a (m x k) matrix.")
        elif len(x.shape) == 2 and (x.shape[0] != m or x.shape[1] > m):
            raise ValueError("state noise covariance must be a (m x k) matrix, "
                             "where m is the number of states.")
        elif len(x.shape) == 3 and (x.shape[0] != m or x.shape[1] > m) and x.shape[-1] != n:
            raise ValueError("state noise covariance must be a  (m x k x n) array, "
                                     "where k <= m is the number of disturbances eta, and m is the number of states.")
    except:
        raise ValueError('The dimension of state noise covariance must be: 1, 2, 3. Warning: np.array(scalar) has dim 0.')

    return x


#### Check system matrix

def check_obs_mtx(x, p, n, m):
    """
    Check system matrix Z of the observation equation.
    Args:
        x (np.array): observation matrix.  Either a vector of length 1 x m,
        a 1 x m x n matrix, p x m matrix or p x m x n
        p: dim of time series (multivariate)
        n: time length
        m: dimension of state

    Returns:

    """
    #TODO unit test for p <1
    flag = None
    if type(x) in [float, int] and m == 1:
        return np.reshape(np.array(x), (1, 1))
    elif not isinstance(x, np.ndarray):
        flag = 1
    if p == 1:
        if not x.shape and m == 1: return np.reshape(x, (1, 1))
        if not x.shape and m != 1: flag = 2
        if len(x.shape) == 1 and m == 1: return x[None]
        if len(x.shape) == 1 and m != 1: flag = 2
        if len(x.shape) == 2:
            if not (x.shape[0] == m and x.shape[1] == n) or (x.shape[0] == m and x.shape[1] == 1)\
                or (x.shape[0] == 1 and x.shape[1] == m): flag = 2
            if not flag and (x.shape[0] == m and x.shape[1] == 1):
                return x.transpose(1, 0)
            if not flag and (x.shape[0] == m and x.shape[1] == n):
                return x[None]
        if len(x.shape) == 3:
            if x.shape[-1] != n: flag = 2
            if not flag and (x.shape[0] == m and x.shape[1] == 1):
                return x.transpose((1, 0, 2))
    else:  # multivariate
        if type(x) in [float, int] and m == 1:
            return np.array(x)[None]
        elif not isinstance(x, np.ndarray):
            flag = 1
        if not x.shape or x.size == 1 or len(x.shape) == 1: flag = 3
        if len(x.shape) == 3:
            if not x.shape in ((p, m), (m, p)): flag = 3 # shape is 2
            elif x.shape == (m, p): return x.transpose((1, 0))
            else: return x
        if len(x.shape) == 3: # p m 1,
            if x.shape[-1] == 1: return np.squeeze(x, axis=-1)
            if x.shape[-1] != n: flag = 3
            if not x.shape in ((p, m, n), (m, p, n)): flag = 3 # shape is 2
            elif x.shape == (m, p, n): return x.transpose((1, 0))
            else: return x

    if flag == 1:
        raise ValueError("obs matrix must be numeric.")
    elif flag == 2:
        raise ValueError("obs matrix must be a (1 x m) or (1 x m x n) matrix,"
                         " where m is the number of states and n is the length of the series.")
    elif flag == 3:
        raise ValueError("obs matrix must be a (p x m) matrix or (p x m x n) array "
                         "where p is the feature of series, m is the number of states, "
                         "and n is the length of the series.")

    raise ValueError(" shape error of obs matrix")


def check_state_mtx(x, m, n):
    """
    Check system matrix T of the state equation. Either a m x m matrix or a m x m x n array.
    Args:
        x (np.array): state matrix
        m:
        n:

    Returns:
        x: m x m  or m x m x n

    """
    flag = None
    if type(x) in [float, int] and m == 1:
        return np.reshape(np.array(x), (1, 1))
    elif not isinstance(x, np.ndarray): flag = 1
    if len(x.shape) == 1 and m!=1: flag = 2
    if len(x.shape) == 1 and m==1: return x[None]
    if len(x.shape) == 2:
        if x.shape[0] != m or x.shape[1] != m: flag = 3
        else: return x
    if len(x.shape) == 3:
        # TODO: unittest: for m != 1 should also np.squeeze(x, axis=-1)
        if x.shape[-1] == 1: return np.squeeze(x, axis=-1)
        if x.shape != (m, m, n): flag = 4
        else: return x

    if flag == 1:
        raise ValueError("state matrix must be numeric.")
    elif flag == 2:
        ValueError("state matrix is a scalar now, it must be a (m x m) matrix.")
    elif flag == 3:
        raise ValueError("state matrix must be a (m x m) matrix, "
                                     "where m is the number of states.")
    elif flag == 4:
        raise ValueError("state matrix must be a (m x m x n) array, "
                                     "where m is the number of states.")

    raise ValueError(" shape error of obs matrix")




###### Check Prior mean and covariance

def check_prior_mean(x, m):
    """
    Check prior mean for the initial state as a vector of length m.
    Args:
        x:
        m:

    Returns:

    """
    if x is None:
        x = np.zeros(m)
    else:
        if type(x) in [float, int]:
            x = np.array(x)
        elif not isinstance(x, np.ndarray):
            raise ValueError("prior mean must be numeric.")
        if x.size == 1:
            #TODO:very confusing in R code check_a1, why repeat when x.size==m
            x = np.repeat(x, m)
        elif x.size != m or len(x.shape)>1:
            raise ValueError("Misspecified prior mean, argument prior mean must be a vector of length m,"
                             " where m is the number of state_names and 1 <= t <= m.")
    return x


def check_prior_cov(x, m):
    """
     Check prior covariance for the initial state as m x m matrix.
    Args:
        x:
        m:

    Returns:

    """
    if x is None:
        x = np.zeros((m, m))
    else:
        if type(x) in [float, int]:
            x = np.array(x)
        elif not isinstance(x, np.ndarray):
            raise ValueError("prior covariance must be numeric.")

        if x.size == 1 and m == 1:
            x = np.reshape(x, (1, 1))
        else:
            if x.shape != (m, m):
                raise ValueError(
                    "Argument prior covariance must be an (m x m) matrix, "
                    "where m is the number of states.")

    return x

# Check Regression Model

def check_xreg(x, n):
    """Check matrix containing covariates with number of rows matching the length of observation y.
    Args:
        x:
        n:

    Returns:

    """
    if x.shape[0] not in (1, n):
        raise ValueError("Number of rows in xreg is not equal to the length of the series y.")
    if not np.isfinite(x).all():
        raise ValueError("Argument xreg must contain only finite values.")


def check_beta(x, k):
    """
    Check A prior for the regression coefficients.
    Should be a vector of prior function (in case of multiple coefficients) or missing in case of no covariates.
    Args:
        x:
        k:

    Returns:

    """
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError("'beta' must be numeric.")
    if len(x) != k:
        raise ValueError("Number of coefficients in beta is not equal to the number of columns of xreg.")
    if not np.isfinite(x).all():
        raise ValueError("Argument 'beta' must contain only finite values.")


def check_positive_const(x, y, multivariate=False):
    """
    Check the matrix of positive constants u for non-Gaussian models
     (of same dimensions as y).
    Args:
        x:
        y:
        multivariate:

    Returns:

    """
    if x is None:
        x = np.ones((y.shape[0]))
    if (x < 0).any():
        raise ValueError("All values of 'u' must be non-negative.")
    if multivariate:
        if len(x) == 1:
            x = np.ones(y.shape)
        if not np.issubdtype(x.dtype, np.number):
            raise ValueError("Argument 'u' must be a numeric matrix.")
        if not np.array_equal(np.shape(y), np.shape(x)):
            raise ValueError("Dimensions of 'y' and 'u' do not match.")
    else: # univariate
        if len(x) == 1:
            x = np.repeat(x, len(y))
        if not np.issubdtype(x.dtype, np.number):
            raise ValueError("Argument 'u' must be a numeric vector.")
        if len(x) != len(y):
            raise ValueError("Lengths of 'u' and 'y' do not match.")
    if not np.isfinite(x).all():
        raise ValueError("Argument 'u' must contain only finite values.")

    return x


def create_regression(beta, xreg, n):
    """

    Args:
        beta: an object belong to Prior Class
        xreg:
        n:

    Returns:

    """

    if xreg is None:
        return {'xreg': np.zeros((0, 0)), 'coefs': np.array([0]), 'beta': None}
    else:
        if beta is None:
            raise ValueError("No prior defined for beta.")
        else:
            if not (is_prior(beta) or is_prior_list(beta)):
                raise ValueError("Prior for beta must belong to 'bssm_prior' or 'bssm_prior_list.")
            if xreg.shape == (len(xreg),):
                if len(xreg) == n:
                    xreg = xreg.reshape((n, 1))
                else:
                    raise ValueError("Length of xreg is not equal to the length of the series y.")

            check_xreg(xreg, n)
            nx = xreg.shape[1]

            if nx == 1 and is_prior_list(beta):
                beta = beta[0]

            if nx > 1:
                coefs = np.array([b['init'] for b in beta])
            else:
                coefs = beta.init

            check_beta(coefs, nx)

            #TODO: In R need to assign name

        return {'xreg': xreg, 'coefs': coefs, 'beta': beta}


# Check model inference
def check_missingness(x):
    """Check the misingness arguments for the mcmc
    Args:
        x: bssm_model
    """
    if not x.model_name in ["ssm_nlg", "ssm_sde"]:
        if not hasattr(x, "prior_parameters"):
            for attribute, value in x.__dict__.items():
                if attribute not in ["y", "update_fn", "prior_fn"] and np.isnan(value):
                    raise ValueError("Missing values not allowed in the model object "
                                     "(except in component 'y').")
        else:
            for attribute, value in x.__dict__.items():
                if attribute not in ["y", "prior_parameters"] and np.isnan(value):
                    raise ValueError("Missing values not allowed in the model object "
                                     "(except in components 'y' and 'prior_parameters').")


def check_intmax(x, postivive=True, max_val=1e5):
    """Chcek whether is non-negative/negative integer

    Args:
        x: input scalar

    Returns:

    """
    if not (isinstance(x, int) and
            (x >= 0 if postivive else x < 0)):
        raise ValueError(f"should be a {str('non-negative') if postivive else str('non-positive')} "
                         f"integer.")
    if x > max_val:
        raise ValueError('The input number exceeds the maximum range.')
