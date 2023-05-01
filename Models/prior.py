import numpy as np

class Prior(object):
    """Define Class object for prior
    Args:
        priorlabel: the type of the prior function
        init: Initial value for the parameter, used in initializing the model
            components and as a starting values in MCMC.
        min_val: Lower bound of the uniform and truncated normal prior.
        max_val: Upper bound of the uniform and truncated normal prior.
        sd: Positive value defining the standard deviation of the (underlying i.e. non-truncated) Normal distribution.
        mean: Mean of the Normal prior.
        shape: Positive shape parameter of the Gamma prior.
        rate: Positive rate parameter of the Gamma prior.

    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        if not getattr(self, 'distribution'):
            raise ValueError('Prior distribution is needed!')

        def_prior = getattr(self, f"{kwargs['distribution']}_prior")
        def_prior()

    def normal_prior(self):
        """Normal Prior
        """
        for attr in ['init', 'mean', 'sd']:
            if not getattr(self, attr):
                raise AttributeError(f'No attritube {attr} found!')
        if not (isinstance(self.init, list) and
                isinstance(self.mean, list) and isinstance(self.sd, list)):
            raise TypeError("Parameters must be passed in list.")
        if not ( all(isinstance(y, (int, float)) for y in self.init) and
                all(isinstance(y, (int, float)) for y in self.mean)
                 and  all(isinstance(y, (int, float)) for y in self.sd) ):
            raise ValueError("Parameters for priors must be numeric.")

        if any(x < 0 for x in self.sd):
            raise ValueError("Standard deviation parameter for Normal distribution must be non-negative.")

        self.prior_distribution = "normal"

        n = max(len(self.init), len(self.mean), len(self.sd))
        if n == 1:
            self.priorlabel = "bssm_prior"
            prior_list = {"init": self.init,
                          "mean": self.mean,
                          "sd": self.sd}
        else:
            self.priorlabel = "bssm_prior_list"
            prior_list = [{"prior_distribution": self.prior_distribution,
                           "init": self.save_pick(self.init, i),
                           "mean": self.save_pick(self.mean, i),
                           "sd": self.save_pick(self.sd, i)} for i in range(n)]
        self.prior = prior_list

    def uniform_prior(self):
        """Uniform Prior
        """
        for attr in ['init', 'min_val', 'max_val']:
            if not getattr(self, attr):
                raise AttributeError(f'No attritube {attr} found!')

        if not (isinstance(self.init, list) and
                isinstance(self.min_val, list) and isinstance(self.max_val, list)):
            raise TypeError("Parameters must be passed in list.")
        if not ( all(isinstance(y, (int, float)) for y in self.init) and
                all(isinstance(y, (int, float)) for y in self.min_val)
                 and all(isinstance(y, (int, float)) for y in self.max_val) ):
            raise ValueError("Parameters for priors must be numeric.")

        if any(x > y for x, y in zip(self.min_val, self.max_val)):
            raise ValueError("Lower bound of uniform distribution must be smaller than upper bound.")
        if any(x < y or x > z for x, y, z in zip(self.init, self.min_val, self.max_val)):
            raise ValueError("Initial value for parameter with uniform prior is not in the support of the prior.")

        self.prior_distribution = "uniform"

        n = max(len(self.init), len(self.min_val), len(self.max_val))
        if n == 1:
            self.priorlabel = "bssm_prior"
            prior_list = {"init": self.init,
                          "min_val": self.min_val,
                          "max_val": self.max_val}
        else:
            self.priorlabel = "bssm_prior_list"
            prior_list = [{"prior_distribution": self.prior_distribution,
                           "init": self.save_pick(self.init, i),
                           "min_val": self.save_pick(self.min_val, i),
                           "max_val": self.save_pick(self.max_val, i)} for i in range(n)]
        self.prior = prior_list

    def halfnormal_prior(self):
        """Halfnormal Prior
        """
        for attr in ['init', 'sd']:
            if not getattr(self, attr):
                raise AttributeError(f'No attritube {attr} found!')
        if not (isinstance(self.init, list) and
                isinstance(self.sd, list)):
            raise TypeError("Parameters must be passed in list.")
        if not ( all(isinstance(y, (int, float)) for y in self.init) and
                all(isinstance(y, (int, float)) for y in self.sd) ):
            raise ValueError("Parameters for priors must be numeric.")

        if any(x < 0 for x in self.sd):
            raise ValueError("Standard deviation parameter for Normal distribution must be non-negative.")

        if any(x < 0 for x in self.init):
            raise ValueError("Initial value for parameter with half-Normal prior must be non-negative.")

        self.prior_distribution = "halfnormal"

        n = max(len(self.init), len(self.sd))
        if n == 1:
            self.priorlabel = "bssm_prior"
            prior_list = {"init": self.init,
                          "sd": self.sd}
        else:
            self.priorlabel = "bssm_prior_list"
            prior_list = [{"prior_distribution": self.prior_distribution,
                           "init": self.save_pick(self.init, i),
                           "sd": self.save_pick(self.sd, i)} for i in range(n)]
        self.prior = prior_list

    def tnormal_prior(self):
        """Truncated normal Prior
        """
        for attr in ['init', 'mean', 'sd', 'min_val', 'max_val']:
            if not hasattr(self, attr):
                if attr == 'min_val':
                    self.min_val = [-np.inf]
                elif attr == 'max_val':
                    self.max_val = [np.inf]
                else:
                    raise AttributeError(f'No attritube {attr} found!')

        if not (isinstance(self.init, list) and
                isinstance(self.mean, list) and isinstance(self.sd, list)):
            raise TypeError("Parameters must be passed in list.")
        if not ( all(isinstance(y, (int, float)) for y in self.init) and
                 all(isinstance(y, (int, float)) for y in self.mean) and
                all(isinstance(y, (int, float)) for y in self.sd) ):
            raise ValueError("Parameters for priors must be numeric.")

        if any(x < y or x > z for x, y, z in zip(self.init, self.min_val, self.max_val)):
            raise ValueError("Initial value for parameter with truncated Normal "
                             "is not between the lower and upper bounds.")
        if any(x < 0 for x in self.sd):
            raise ValueError("Standard deviation parameter for truncated Normal distribution must be positive.")

        self.prior_distribution = "tnormal"

        n = max(len(self.init), len(self.mean), len(self.sd))
        if n == 1:
            self.priorlabel = "bssm_prior"
            prior_list = {"init": self.init,
                          "mean": self.mean,
                          "sd": self.sd,
                          "min_val": self.min_val,
                          "max_val": self.max_val}
        else:
            self.priorlabel = "bssm_prior_list"
            prior_list = [{"prior_distribution": self.prior_distribution,
                           "init": self.save_pick(self.init, i),
                           "mean": self.save_pick(self.mean, i),
                           "sd": self.save_pick(self.sd, i),
                           "min_val": self.save_pick(self.min_val, i),
                           "max_val": self.save_pick(self.max_val, i)} for i in range(n)]
        self.prior = prior_list

    def gamma_prior(self):
        """Gamma Prior
        """
        for attr in ['init', 'shape', 'rate']:
            if not getattr(self, attr):
                raise AttributeError(f'No attritube {attr} found!')
        if not (isinstance(self.init, list) and
                isinstance(self.shape, list) and isinstance(self.rate, list)):
            raise TypeError("Parameters must be passed in list.")
        if not ( all(isinstance(y, (int, float)) for y in self.init) and
                 all(isinstance(y, (int, float)) for y in self.shape) and
                all(isinstance(y, (int, float)) for y in self.rate) ):
            raise ValueError("Parameters for priors must be numeric.")

        if any(x < 0 for x in self.shape):
            raise ValueError("Shape parameter for Gamma distribution must be positive.")
        if any(x < 0 for x in self.rate):
            raise ValueError("Rate parameter for Gamma distribution must be positive.")

        self.prior_distribution = "gamma"

        n = max(len(self.init), len(self.shape), len(self.rate))
        if n == 1:
            self.priorlabel = "bssm_prior"
            prior_list = {"init": self.init,
                          "shape": self.shape,
                          "rate": self.rate}
        else:
            self.priorlabel = "bssm_prior_list"
            prior_list = [{"prior_distribution": self.prior_distribution,
                           "init": self.save_pick(self.init, i),
                           "shape": self.save_pick(self.shape, i),
                           "rate": self.save_pick(self.rate, i)} for i in range(n)]
        self.prior = prior_list

    def _toR(self):
        """Transfer the prior object to R structure

        Returns: a list in R (Dict in Python)
        """
        prior_dict = []
        if isinstance(self.prior, list): #prior_list
            for idx in range(len(self.prior)):
                sub_list = []
                for key, value in self.prior[idx].items():
                    sub_list.append(value)
                prior_dict.append(sub_list)
        elif isinstance(self.prior, dict):
            prior_dict.append(self.prior_distribution)
            for key, value in self.prior.items():
                prior_dict.append(value[0])
        else:
            raise TypeError('No matching type for prior!')

        return prior_dict

    @staticmethod
    def save_pick(x, i):
        """Pick the element at index i given the input x.
        Args:
            x:
            i:

        Returns: the element at index i/ or the last element given the input x.

        """
        return x[min(len(x) -1, i)]


def is_prior_list(prior_object):
    """Check whether the prior has multiple types, for multivariate time series

    Args:
        prior_object: prior function

    Returns:
        True if has multiple types of prior functions else False
    """
    return True if 'bssm_prior_list' in prior_object.priorlabel else False


def is_prior(prior_object):
    """Check whether the prior belongs to a prior

    Args:
        prior_object: prior function

    Returns:
        True if has multiple types of prior functions else False
    """
    return True if 'bssm_prior' in prior_object.priorlabel else False


def combine_prior(x):
    """Combine all the prior objects
    Args:
        x:

    Returns:

    """
    pass