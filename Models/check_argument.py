import numpy as np

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
                if (finite_x != finite_x.astype(int)).any():
                    raise ValueError(f"Non-integer values not allowed for {distribution} distribution.")
        if np.isinf(x).any():
            raise ValueError("Argument 'y' must contain only finite or NA values.")
    return x