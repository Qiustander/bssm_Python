import numpy as np
import tensorflow as tf


def check_y(x, fill_missing=0., filling=True):
    """
    Args:
        x: input observed time series with dimension n x p
        can contain missing values for imputation
        fill_missing: the values filled for NA positions
    Returns:
        shape_x (tuple): shape of time series n, p
        x: time series n x p
    """

    if type(x) in [float, int]:
        raise TypeError("Scalar is not allowed.")
    elif type(x) in [list, tuple]:
        x = np.array(x)
    if isinstance(x, tf.Tensor):
        x = x.numpy()
    if np.isinf(x).any():
        raise TypeError("Argument 'y' must contain only finite or NA values.")
    if not np.isnan(x).all():
        x = np.nan_to_num(x)
        if len(x.shape) > 3:
            raise ValueError("Only two-dimensional time series n x p is accepted.")
        if len(x.shape) == 2 and x.shape[0] < 2:
            raise ValueError("Number of rows in 'y', i.e. number of time points, must be > 1.")
        if len(x.shape) == 1:
            if x.shape[0] < 2:
                raise ValueError("Length of argument y, i.e. number of time points, must be > 1.")
            x = x[..., None]
    else:
        raise TypeError("Could not be empty array, or could not be all NA values.")
    return x.shape, x


def check_mu(x):

    if isinstance(x, np.ndarray) and len(x) == 1:
        return x
    if type(x) in [float, int, list, np.float32, np.float64]:
        return np.array(x)[None]
    if not x.shape:
        x = x[None]
    if len(x) != 1:
        raise ValueError("Argument 'mu' must be of length one.")
    if not np.isfinite(x).all():
        raise ValueError("Argument 'mu' must contain only finite values.")
    raise ValueError("No return.")


def check_rho(x):
    """

    Args:
        x: rho parameter for AR(1) model

    Returns:
        x: if it satisfies the condition
    """
    if not x:
        raise ValueError("rho could not be None.")
    try:
        if type(x) in [float, int]:
            x = np.array(x)[None]
        elif type(x) in [list, tuple]:
            x = np.array(x)
        elif isinstance(x, tf.Tensor):
            x = x.numpy()
    except:
        raise TypeError("Must be numeric")
    if len(x) != 1:
        raise ValueError("Argument 'rho' must be of length one.")
    if abs(x) >= 1:
        raise ValueError("Argument 'rho' must be strictly between -1 and 1.")

    if isinstance(x, np.ndarray) and len(x.shape) == 1:
        return x
    if not x.shape:
        return x[None]


def check_phi(x):
    if x < 0:
        raise ValueError("Parameter 'phi' must be non-negative.")


def check_prop(x):
    if not type(x) in [np.ndarray, float, int] or x >= 1 or x <= 0:
        raise ValueError("Argument must be on interval (0, 1).")


##### Check input C & D

def check_input_obs(x, p, n):
    """Check intercept terms \eqn{D_t} for the observations equation,
    given as a [p,] vector, or p x n (time-varying) matrix.
    Args:
        x:
        p:
        n:

    Returns:

    """
    if x is None:
        return np.zeros([p, ])
    else:
        if type(x) in [float, int, list, np.float32, np.float64]:
            x = np.array(x)
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        elif not isinstance(x, np.ndarray):
            raise ValueError("input obs must be numeric.")
        x = x.squeeze()
        if p == 1:  # univariate
            if not x.shape:
                return x[None]
            if not np.size(x) in (1, n):
                raise ValueError("input obs variable must be a scalar or length n,"
                                 " where n is the number of observations.")
        else:  # multivariate
            if not x.shape:
                return x[None]
            # if x.shape == 0 and np.abs(x - 0) <1e-8:
            #     return np.array([x]*p)
            if len(x.shape) == 1 and not x.shape[0] == p:
                raise ValueError("input obs variable must be p or p x n matrix, "
                                 "where p is the number of series.")
            elif len(x.shape) == 2 and (not x.shape[-1] == n or not x.shape[0] == p):
                raise ValueError("input obs variable must be p or p x n matrix, "
                                 "where p is the number of series.")
    return x


def check_input_state(x, m, n):
    """Check Intercept terms \eqn{C_t} for the state equation,
    given as a [m,] vector, or m x n (time-varying) matrix.
    Args:
        x:
        m:
        n:

    Returns:

    """
    if x is None:
        return np.zeros([m, ])
    else:
        if type(x) in [float, int, list, np.float32, np.float64]:
            x = np.array(x)
        if isinstance(x, tf.Tensor):
            x = x.numpy()
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
        Either a p x p matrix or p x p x n (time-varying) array.
    Args:
        x:
        p:
        n:

    Returns:

    """
    if n == 1:
        raise ValueError("Length of time series n must larger than 1.")
    if type(x) in [float, int, list, np.float32, np.float64]:
        x = np.array(x)
    if isinstance(x, tf.Tensor):
        x = x.numpy()
    elif not isinstance(x, np.ndarray):
        raise ValueError("obs noise std matrix must be numeric.")
    x = x.squeeze()
    if p == 1:
        if not x.shape:
            return x[None, None]
        if (x.size > 1 and x.size != n):
            raise ValueError("time-varying obs noise std matrix must be p x p x n, "
                             "where n is the length of the time series y")
        else:
            return x[None, None]
    else:  # multivariate
        if not x.shape or x.shape == 1:
            raise ValueError(
                "obs noise std matrix must be a m x m matrix ")
        if len(x.shape) == 2:
            if x.shape != (p, p):
                raise ValueError(
                    "obs noise std matrix must be a p x p matrix ")
        elif len(x.shape) == 3:
            if x.shape != (p, p, n):
                raise ValueError(
                    "obs noise std matrix must p x p x n matrix, "
                    "where p is the number of series and n is the length of the series.")

    return x


def check_state_noise(x, m, n):
    """
    Check noise  matrix R (lower triangular) the state equation.
    Either a m x k (k<=m) matrix or a m x k (k<=m) x n (time-varying) array.
    Args:
        x:
        m:
        n:

    Returns:

    """
    if n == 1:
        raise ValueError("Length of time series n must larger than 1.")
    if type(x) in [float, int, list, np.float32, np.float64]:
        x = np.array(x)
    if isinstance(x, tf.Tensor):
        x = x.numpy()
    elif not isinstance(x, np.ndarray):
        raise ValueError("state noise must be numeric.")
    x = x.squeeze()

    if m == 1:
        if not x.shape:
            return x[None, None]  # [[scalar]]
        if (x.size > 1 and x.size != n):
            raise ValueError("time-varying state noise std matrix must be m x k (k<=m) x n, "
                             "where n is the length of the time series y")
        else:
            return x[None, None]
    else:
        if not x.shape or x.shape == 1:
            raise ValueError(
                "state noise std matrix must be a m x x k (k<=m) matrix ")
        if len(x.shape) == 2:
            if x.shape[0] != m or x.shape[-1] > m:
                raise ValueError(
                    "state noise std matrix must be a m x k (k<=m) matrix ")
        elif len(x.shape) == 3:
            if x.shape[0] != m or x.shape[1] > m or x.shape[-1] != n:
                raise ValueError(
                    "state noise std matrix must m x k (k<=m) x n matrix, "
                    "where p is the number of series and n is the length of the series.")

    return x


#### Check system matrix

def check_obs_mtx(x, p, n, m):
    """
    Check system matrix Z of the observation equation.
    Args:
        x (np.array): observation matrix.  Either a  p x m matrix or p x m x n
        p: dim of time series (multivariate)
        n: time length
        m: dimension of state

    Returns:

    """

    def error_case(case):
        if case == 1:
            return "obs matrix must be numeric."
        elif case == 2:
            return "obs matrix must be a (p x m) matrix or (p x m x n) array " \
                   "where p is the feature of series, m is the number of states, " \
                   "and n is the length of the series."

    if type(x) in [float, int, list, np.float32, np.float64]:
        if m == 1:
            x = np.array(x)
        else:
            raise ValueError(error_case(2))
    if isinstance(x, tf.Tensor):
        x = x.numpy()
    elif not isinstance(x, np.ndarray):
        raise TypeError(error_case(1))
    if len(x.shape) == 3 and x.shape[-1] == 1:  # p m 1,
        x = np.squeeze(x, axis=-1)
    if len(x.shape) == 3:  # still have 3 dim
        if x.shape == (p, m, n):
            return x
        else:
            raise ValueError(error_case(2))

    if p == 1:
        if not x.shape and m == 1: return np.reshape(x, (1, 1))
        if not x.shape and m != 1: raise ValueError(error_case(2))
        if len(x.shape) == 1 and m == 1: return x[None]
        if len(x.shape) == 1 and m != 1: raise ValueError(error_case(2))
        if len(x.shape) == 2:
            if not x.shape in ((p, m), (p, n), (m, n)): raise ValueError(error_case(2))
            # if not ((x.shape[0] == m and x.shape[1] == n) or (x.shape[0] == m and x.shape[1] == 1)
            #     or (x.shape[0] == 1 and x.shape[1] == m)): raise ValueError(error_case(2))
            if (x.shape[0] == m and x.shape[1] == n and p == 1):
                return x[None]
            else:
                return x
    else:  # multivariate
        if not x.shape or x.size == 1 or len(x.shape) == 1: raise ValueError(error_case(2))
        if m == 1 and len(x.shape) == 1: return np.array(x)[None]
        if x.shape == (p, m): return x

    raise ValueError("shape error of obs matrix")


def check_state_mtx(x, m, n):
    """
    Check system matrix T of the state equation. Either a m x m matrix or a m x m x n (time-varying) array.
    Args:
        x (np.array): state matrix
        m:
        n:

    Returns:
        x: m x m  or m x m x n

    """

    def error_case(case):
        if case == 1:
            return "state matrix must be numeric."
        elif case == 2:
            return "state matrix can not be a scalar, it must be a (m x m) matrix."
        elif case == 3:
            return "state matrix must be a (m x m) matrix, " \
                   "where m is the number of states."
        elif case == 4:
            return "state matrix must be a (m x m x n) array, " \
                   "where m is the number of states."

    if type(x) in [float, int, list, np.float32, np.float64]:
        if m == 1:
            x = np.reshape(np.array(x), (1, 1))
        else:
            raise ValueError(error_case(2))
    if isinstance(x, tf.Tensor):
        x = x.numpy()
    elif not isinstance(x, np.ndarray):
        raise TypeError(error_case(1))
    if len(x.shape) == 1 and m != 1: raise ValueError(error_case(2))
    if len(x.shape) == 1 and m == 1: return x[None]
    if len(x.shape) == 2:
        if m == 1 and x.shape == (m, n): return x[None]
        if x.shape != (m, m):
            raise ValueError(error_case(3))
        else:
            return x
    if len(x.shape) == 3:
        if x.shape[-1] == 1 and x.shape[:-1] == (m, m): return np.squeeze(x, axis=-1)
        if x.shape != (m, m, n):
            raise ValueError(error_case(4))
        else:
            return x

    raise ValueError(" shape error of obs matrix")


###### Check Prior mean and covariance

def check_prior_mean(x, m):
    """
    Check prior mean for the initial state as a vector of length [m,].
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
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        elif not isinstance(x, np.ndarray):
            raise TypeError("prior mean must be numeric.")
        if x.size == 1:
            x = np.repeat(x, m)
        elif x.size != m or len(x.shape) > 1:
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
        if type(x) in [float, int, list, np.float32, np.float64]:
            x = np.array(x)
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        elif not isinstance(x, np.ndarray):
            raise TypeError("prior covariance must be numeric.")

        if x.size == 1 and m == 1:
            x = np.reshape(x, (1, 1))
        else:
            if x.shape != (m, m):
                raise ValueError(
                    "Argument prior covariance must be an (m x m) matrix, "
                    "where m is the number of states.")

    return x

