import pytest
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from Models.prior import *
from numpy.testing import assert_array_almost_equal_nulp
import os.path as pth
import os


# Automatic convertion between R and Python objects
numpy2ri.activate()

base = importr('base', lib_loc="/usr/lib/R/library")
bssm_package = importr('bssm', lib_loc="/home/stander/R/x86_64-pc-linux-gnu-library/4.2")

# Import R function
ro.r("""source('{path_name}')""".
     format(path_name=pth.join(pth.abspath(pth.join(os.getcwd(), os.pardir, os.pardir)), 'bssm_R/R/priors.R')))
r_normal = ro.globalenv['normal_prior']
r_uniform = ro.globalenv['uniform_prior']
r_halfnormal = ro.globalenv['halfnormal_prior']
r_tnormal = ro.globalenv['tnormal_prior']
r_gamma = ro.globalenv['gamma_prior']
r_combine_prior = ro.globalenv['combine_priors']
r_is_prior = ro.globalenv['is_prior']
r_is_prior_list = ro.globalenv['is_prior_list']


class TestPrior:
    """
    Test Prior distribution
    """

    def test_normal_prior(self):
        # missing
        with pytest.raises(AttributeError):
            Prior(init=[0], mean=[1], distribution='normal')
        with pytest.raises(AttributeError):
            Prior(init=[0], mean=[1], sd=[1])
        # not list
        with pytest.raises(TypeError):
            Prior(init=[0], mean=[1], sd=1, distribution='normal')
        # not numeric
        with pytest.raises(ValueError):
            Prior(init=[0], mean=[1], sd=[1,'s'], distribution='normal')
        # sd <0
        with pytest.raises(ValueError):
            Prior(init=[0], mean=[1], sd=[-1], distribution='normal')
        # check bssm_prior
        prior_obj = Prior(init=[0], mean=[1], sd=[1], distribution='normal')
        assert prior_obj.priorlabel == "bssm_prior"
        assert type(prior_obj.prior) == dict
        # check bssm_prior_list
        prior_obj = Prior(init=[0], mean=[1], sd=[1, 1], distribution='normal')
        assert prior_obj.priorlabel == "bssm_prior_list"
        assert type(prior_obj.prior) == list
        assert type(prior_obj.prior[0]) == dict

        # Execute R code
        r_result = r_normal('init = c(0.1, 2.5), mean = 0.1, sd = c(1.5, 2.8)')

    def test_uniform_prior(self):
        # missing
        with pytest.raises(AttributeError):
            Prior(init=[0], min_val=[1], distribution='uniform')
        with pytest.raises(AttributeError):
            Prior(init=[0], min_val=[1], max_val=[1])
        # not list
        with pytest.raises(TypeError):
            Prior(init=[0], min_val=[1], max_val=1, distribution='uniform')
        # not numeric
        with pytest.raises(ValueError):
            Prior(init=[0], min_val=[1], max_val=[1,'s'], distribution='uniform')
        # min max
        with pytest.raises(ValueError):
            Prior(init=[3], min_val=[2], max_val= [1], distribution='uniform')
        with pytest.raises(ValueError):
            Prior(init=[-1], min_val=[0], max_val=[2], distribution='uniform')
        with pytest.raises(ValueError):
            Prior(init=[3], min_val=[0], max_val=[2], distribution='uniform')
        # check bssm_prior
        prior_obj = Prior(init=[1.5], min_val=[1], max_val=[2], distribution='uniform')
        assert prior_obj.priorlabel == "bssm_prior"
        assert type(prior_obj.prior) == dict
        # check bssm_prior_list
        prior_obj = Prior(init=[1.5], min_val=[1], max_val=[2,2], distribution='uniform')
        assert prior_obj.priorlabel == "bssm_prior_list"
        assert type(prior_obj.prior) == list
        assert type(prior_obj.prior[0]) == dict

    def test_halfnormal_prior(self):
        # missing
        with pytest.raises(AttributeError):
            Prior(init=[0], distribution='halfnormal')
        with pytest.raises(AttributeError):
            Prior(init=[0], sd=[1])
        # not list
        with pytest.raises(TypeError):
            Prior(init=[0], sd=1, distribution='halfnormal')
        # not numeric
        with pytest.raises(ValueError):
            Prior(init=[0], sd=[1,'s'], distribution='halfnormal')
        # min max
        with pytest.raises(ValueError):
            Prior(init=[3], sd=[-1], distribution='halfnormal')
        with pytest.raises(ValueError):
            Prior(init=[-1], sd=[1], distribution='halfnormal')
        # check bssm_prior
        prior_obj = Prior(init=[1.5], sd=[1], distribution='halfnormal')
        assert prior_obj.priorlabel == "bssm_prior"
        assert type(prior_obj.prior) == dict
        # check bssm_prior_list
        prior_obj = Prior(init=[1.5], sd=[1, 3], distribution='halfnormal')
        assert prior_obj.priorlabel == "bssm_prior_list"
        assert type(prior_obj.prior) == list
        assert type(prior_obj.prior[0]) == dict

    def test_tnormal_prior(self):
        # missing
        with pytest.raises(AttributeError):
            Prior(init=[0], mean=[1],  distribution='tnormal')
        with pytest.raises(AttributeError):
            Prior(init=[0], sd=[1])
        # not list
        with pytest.raises(TypeError):
            Prior(init=[0], sd=1, mean=[1], distribution='tnormal')
        # not numeric
        with pytest.raises(ValueError):
            Prior(init=[0], sd=[1,'s'], mean=[1], distribution='tnormal')
        # min max
        with pytest.raises(ValueError):
            Prior(init=[3], sd=[1], min_val=[1], max_val= [2], mean=[1], distribution='tnormal')
        with pytest.raises(ValueError):
            Prior(init=[-1], sd=[-1], mean=[1], distribution='tnormal')
        # check bssm_prior
        prior_obj = Prior(init=[1.5], sd=[1], mean=[1], distribution='tnormal')
        assert prior_obj.priorlabel == "bssm_prior"
        assert type(prior_obj.prior) == dict
        # check bssm_prior_list
        prior_obj = Prior(init=[1.5], sd=[1, 3], mean=[1], distribution='tnormal')
        assert prior_obj.priorlabel == "bssm_prior_list"
        assert type(prior_obj.prior) == list
        assert type(prior_obj.prior[0]) == dict

    def test_gamma_prior(self):
        # missing
        with pytest.raises(AttributeError):
            Prior(init=[0], shape=[1],  distribution='gamma')
        with pytest.raises(AttributeError):
            Prior(init=[0], shape=[1], rate=[2])
        # not list
        with pytest.raises(TypeError):
            Prior(init=[0], shape=1, rate=[1], distribution='gamma')
        # not numeric
        with pytest.raises(ValueError):
            Prior(init=[0], shape=[1,'s'], rate=[1], distribution='gamma')
        # min max
        with pytest.raises(ValueError):
            Prior(init=[3], shape=[-1], rate=[2], distribution='gamma')
        with pytest.raises(ValueError):
            Prior(init=[-1], shape=[1], rate=[-2], distribution='gamma')
        # check bssm_prior
        prior_obj = Prior(init=[1.5], shape=[1], rate=[2], distribution='gamma')
        assert prior_obj.priorlabel == "bssm_prior"
        assert type(prior_obj.prior) == dict
        # check bssm_prior_list
        prior_obj = Prior(init=[1.5], shape=[1], rate=[2,3, 7], distribution='gamma')
        assert prior_obj.priorlabel == "bssm_prior_list"
        assert type(prior_obj.prior) == list
        assert type(prior_obj.prior[0]) == dict



