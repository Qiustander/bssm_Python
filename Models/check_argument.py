import numpy as np


class CheckArg:
    '''
    For checking the arguments of the ssm models.
    '''

    def __init__(self):
        pass

    def check_all_argument(self, y):
        """
        Args:
            y: The response time series.
        :return:
        """
        self.check_y(y, multivariate=False, distribution="gaussian")

    @staticmethod
    def check_y(x, multivariate=False, distribution="gaussian"):
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
                    raise ValueError(f"Negative values not allowed for {distribution} distribution.")
                elif distribution in ["negative binomial", "binomial", "poisson"]:
                    finite_x = x[np.isfinite(x)]
                    if np.any((finite_x != finite_x.astype(int))):
                        raise ValueError(f"Non-integer values not allowed for {distribution} distribution.")
            if np.isinf(x).any():
                raise ValueError("Argument 'y' must contain only finite or NA values.")

    @staticmethod
    def check_period(x, n):
        if not isinstance(x, int):
            raise ValueError("Argument 'period' should be a single integer.")
        else:
            if x < 3:
                raise ValueError("Argument 'period' should be an integer larger than 2.")
            if x >= n:
                raise ValueError("Period should be less than the number of time points.")

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def check_xreg(x, n):
        if x.shape[0] not in (0, n):
            raise ValueError("Number of rows in xreg is not equal to the length of the series y.")
        if not np.isfinite(x).all():
            raise ValueError("Argument xreg must contain only finite values.")

    @staticmethod
    def check_beta(x, k):
        if not np.issubdtype(x.dtype, np.number):
            raise ValueError("'beta' must be numeric.")
        if len(x) != k:
            raise ValueError("Number of coefficients in beta is not equal to the number of columns of xreg.")
        if not np.isfinite(x).all():
            raise ValueError("Argument 'beta' must contain only finite values.")

    @staticmethod
    def check_mu(x):
        if len(x) != 1:
            raise ValueError("Argument 'mu' must be of length one.")
        if not np.isfinite(x).all():
            raise ValueError("Argument 'mu' must contain only finite values.")

    @staticmethod
    def check_rho(x):
        if len(x) != 1:
            raise ValueError("Argument 'rho' must be of length one.")
        if abs(x) >= 1:
            raise ValueError("Argument 'rho' must be strictly between -1 and 1.")

    @staticmethod
    def check_phi(x):
        if x < 0:
            raise ValueError("Parameter 'phi' must be non-negative.")

    @staticmethod
    def check_u(x, y, multivariate=False):
        if (x < 0).any():
            raise ValueError("All values of 'u' must be non-negative.")
        if multivariate:
            if len(x) == 1:
                x = np.matrix(x).reshape(np.shape(y))
            if not (isinstance(x, np.matrix) and np.issubdtype(x.dtype, np.number)):
                raise ValueError("Argument 'u' must be a numeric matrix or multivariate ts object.")
            if not np.array_equal(np.shape(y), np.shape(x)):
                raise ValueError("Dimensions of 'y' and 'u' do not match.")
        else:
            if len(x) == 1:
                x = np.repeat(x, len(y))
            if not (isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number)):
                raise ValueError("Argument 'u' must be a numeric vector or ts object.")
            if len(x) != len(y):
                raise ValueError("Lengths of 'u' and 'y' do not match.")
        if not np.isfinite(x).all():
            raise ValueError("Argument 'u' must contain only finite values.")
