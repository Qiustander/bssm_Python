import numpy as np


def check_y(x, multivariate=False, distribution="gaussian"):
    """
    Args:
        x: input time series with length n (second last dimension), can contain missing values for imputation
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
            if not isinstance(x, np.ndarray) or x.ndim > 1:
                if x.ndim == 2 and x.shape[1] == 1 and x.shape[0] < 3:
                    x = x.flatten()
                else:
                    raise ValueError("Argument 'y' must be a Tensor or Numpy array.")
            if len(x) < 2:
                raise ValueError("Length of argument y, i.e. number of time points, must be > 1.")
            if distribution != "gaussian" and (np.logical_not(np.isnan(x)) & (x < 0)).any():
                raise AttributeError(f"Negative values not allowed for {distribution} distribution.")
            elif distribution in ["negative binomial", "binomial", "poisson"]:
                finite_x = x[np.isfinite(x)]
                if np.any((finite_x != finite_x.astype(int))):
                    raise AssertionError(f"Non-integer values not allowed for {distribution} distribution.")
        if np.isinf(x).any():
            raise TypeError("Argument 'y' must contain only finite or NA values.")

    return x.shape


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

    if not isinstance(x, (int, float)):
        raise ValueError(f"Argument {param} must be numeric.")
    if x < 0:
        raise ValueError(f"Argument {param} must be non-negative.")
    if np.isinf(x):
        raise ValueError(f"Argument {param} must be finite.")



def check_mu(x):
    if len(x) != 1:
        raise ValueError("Argument 'mu' must be of length one.")
    if not np.isfinite(x).all():
        raise ValueError("Argument 'mu' must contain only finite values.")


def check_rho(x):
    if len(x) != 1:
        raise ValueError("Argument 'rho' must be of length one.")
    if abs(x) >= 1:
        raise ValueError("Argument 'rho' must be strictly between -1 and 1.")


def check_phi(x):
    if x < 0:
        raise ValueError("Parameter 'phi' must be non-negative.")



def check_prior(x, name):
    if not is_prior(x) or is_prior_list(x):
        raise ValueError(f"{name} must belong 'bssm_prior' or 'bssm_prior_list'.")


def check_prop(x, name="target"):
    if len(x) > 1 or x >= 1 or x <= 0:
        raise ValueError(f"Argument '{name}' must be on interval (0, 1).")


##### Check input C & D

def check_input_obs(x, p, n):
    """Check intercept terms \eqn{D_t} for the observations equation,
        given as a scalar or vector of length n.
    Args:
        x:
        p:
        n:

    Returns:

    """
    if x is None:
        x = 0 if p == 1 else np.zeros([p, 1])
    else:
        if not np.issubdtype(x.dtype, np.number):
            raise ValueError("D must be numeric.")

        if p == 1:
            if not np.size(x) in (1, n):
                raise ValueError("'D' must be a scalar or length n,"
                                 " where n is the number of observations.")
        else:
            if not (x.shape[0] == p and (x.shape[-1] in (1, n))):
                raise ValueError("'D' must be p x 1 or p x n matrix, "
                                 "where p is the number of series.")
    return x


def check_input_state(x, m, n):
    """Check Intercept terms \eqn{C_t} for the state equation,
    given as a m x 1 or m x n matrix.
    Args:
        x:
        m:
        n:

    Returns:

    """
    if x is None:
        x = np.zeros([m, 1])
    else:
        if not np.issubdtype(x.dtype, np.number):
            raise ValueError("C must be numeric.")

        if not (x.shape[0] == m and (x.shape[-1] in (1, n))):
            raise ValueError("'C' must be m x 1 or m x n matrix, "
                             "where m is the number of states.")
    return x


def check_noise_std(x, p, n, multivariate=False):
    """Check vector H of standard deviations of noise. Either a scalar or a vector of  length n,
        or p x p matrix or p x p x n array.
    Args:
        x:
        p:
        n:
        multivariate:

    Returns:

    """
    if not multivariate:
        if (x.size > 1 and x.size != n and len(x.shape) == 1):
            raise ValueError("'H' must be a scalar or length n, "
                             "where n is the length of the time series y")
    else:
        match x.shape:
            case 2:
                if x.shape != (p, p):
                    raise ValueError(
                        "Argument 'H' must be a p x p matrix "
                        "where p is the number of series and n is the length of the series.")
                else:
                    x = np.reshape(x, (p, p, 1))
            case 3:
                if x.shape[:2] != (p, p) and x.shape[2] not in (1, n):
                    raise ValueError(
                        "Argument 'H' must  p x p x n array, "
                        "where p is the number of series and n is the length of the series.")
    return x


#### Check system matrix


def check_obs_mtx(x, p, n, multivariate=False):
    """
    Check system matrix Z of the observation equation.
    Args:
        x (np.array): observation matrix.  Either a vector of length m, a m x n matrix.
        p:
        n:
        multivariate:

    Returns:

    """
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError("'Z' must be numeric.")
    if not multivariate:
        if np.size(x) == 1:  # single variable case
            x = np.reshape(x, (1, 1))
        else:
            if x.shape[1] not in (1, n):
                raise ValueError("'Z' must be a (m x 1) or (m x n) matrix,"
                                 " where m is the number of states and n is the length of the series.")
            else:
                x = np.reshape(x,
                               (x.shape[0], (n - 1) * (np.maximum(x.shape[1], 0, where=~np.isnan(x.shape[1])) > 1) + 1))
    else:  # multivariate
        if x.size == 1:
            x = np.reshape(x, (1, 1, 1))
        else:
            if x.shape[0] != p or x.shape[2] not in (1, n):
                raise ValueError("'Z' must be a (p x m) matrix or (p x m x n) array "
                                 "where p is the feature of series, m is the number of states, "
                                 "and n is the length of the series.")
            x = np.reshape(x,
                           (p, x.shape[1], (n - 1) * (np.maximum(x.shape[2], 0, where=~np.isnan(x.shape[2])) > 1) + 1))

    return x


def check_state_mtx(x, m, n):
    """
    Check system matrix T of the state equation. Either a m x m matrix or a m x m x n array.
    Args:
        x (np.array): state matrix
        m:
        n:

    Returns:
        x: m x m x 1 or m x m x n

    """
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError("'T' must be numeric.")
    try:
        match len(x.shape):
            case 1:
                if m != 1:
                    raise ValueError("'T' is a scalar now, it must be a (m x m) matrix.")
                x = np.reshape(x, (1, 1, 1))
            case 2:
                if x.shape[0] != m or x.shape[1] != m:
                    raise ValueError("'T' must be a (m x m) matrix, "
                                     "where m is the number of states.")
                x = np.reshape(x, (m, m, 1))
            case 3:
                if x.shape[2] not in (1, n):
                    raise ValueError("'T' must be a (m x m x 1) or (m x m x n) array, "
                                     "where m is the number of states.")
                x = np.reshape(x, (m, m, (n - 1) * (np.maximum(x.shape[2], 0, where=~np.isnan(x.shape[2])) > 1) + 1))
    except:
        raise ValueError('The dimension of T must be: 1, 2, 3. Warning: np.array(scalar) has dim 0.')

    return x


def check_mtx_lower(x, m, n):
    """
    Check lower triangular matrix R the state equation. Either a m x k matrix or a m x k x n array.
    Args:
        x:
        m:
        n:

    Returns:

    """
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError("'R' must be numeric.")
    try:
        match len(x.shape):
            case 1:
                if m != 1:
                    raise ValueError("'R' is a scalar now, it must be a (m x k) matrix.")
                x = np.reshape(x, (1, 1, 1))
            case 2:
                if x.shape[0] != m or x.shape[1] > m:
                    raise ValueError("R must be a (m x k) matrix, "
                                     "where m is the number of states.")
                x = np.reshape(x, (m, x.shape[1], 1))
            case 3:
                if x.shape[2] not in (1, n):
                    raise ValueError("R must be a (m x k) matrix, (m x k x 1), or (m x k x n) array, "
                                     "where k <= m is the number of disturbances eta, and m is the number of states.")
                x = x.reshape(m, x.shape[1], n if x.shape[2] > 1 else 1)
    except:
        raise ValueError('The dimension of T must be: 1, 2, 3. Warning: np.array(scalar) has dim 0.')

    return x


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
        if not np.issubdtype(x.dtype, np.number):
            raise ValueError("'Prior mean' must be numeric.")
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
        if not np.issubdtype(x.dtype, np.number):
            raise ValueError("'Prior Covariance' must be numeric.")

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
                coefs = beta['init']

            check_beta(coefs, nx)

            #TODO: In R need to assign name

        return {'xreg': xreg, 'coefs': coefs, 'beta': beta}
