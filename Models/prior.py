import numpy as np


class Prior(object):
    """Define Class object for prior

    """
    def __init__(self):
        self.priorlabel = {}


def is_prior_list(prior_object):
    """Check whether the prior has multiple types, for multivariate time series

    Args:
        prior_object: prior function

    Returns:
        True if has multiple types of prior functions else False
    """
    return True if 'bssm_prior_list' in prior_object.priorlabel else False