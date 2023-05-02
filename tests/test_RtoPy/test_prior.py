import pytest
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from Models.prior import *
from numpy.testing import assert_array_almost_equal_nulp
import os.path as pth
import os


# Automatic convertion between R and Python objects
numpy2ri.activate()
pandas2ri.activate()

base = importr('base', lib_loc="/usr/lib/R/library")
bssm_package = importr('bssm', lib_loc=f"{pth.expanduser('~')}/R/x86_64-pc-linux-gnu-library/4.2")

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

################### Test Normal ##################################
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

    def test_normal_with_r(self):

        # # prior_list
        # r_result1 = r_normal(init = ro.r('c(0.1, 2.5)'), mean = 0.1, sd = ro.r('c(1.5, 2.8)'))
        # py_obj1 = [[element for element in sublist] for sublist in r_result1]
        # prior_obj = Prior(init=[0.1, 2.5], mean=[0.1], sd=[1.5, 2.8], distribution='normal')._toR()
        # for idx in range(len(prior_obj)):
        #     for x, y in zip(prior_obj[idx], py_obj1[idx]):
        #         assert x == y[0]
        #
        # # prior
        # r_result2 = r_normal(init = 2.5, mean = 0.1, sd = 2.8)
        # py_obj2 = [[element for element in sublist] for sublist in r_result2]
        # prior_obj = Prior(init=[2.5], mean=[0.1], sd=[2.8], distribution='normal')._toR()
        # for x, y in zip(prior_obj, py_obj2):
        #     assert x == y[0]

        # # prior_list
        r_result1 = r_normal(init = ro.r('c(0.1, 2.5)'), mean = 0.1, sd = ro.r('c(1.5, 2.8)'))
        prior_obj = Prior(init=[0.1, 2.5], mean=[0.1], sd=[1.5, 2.8], distribution='normal')._toR()
        comp_result = base.all_equal(prior_obj, r_result1)
        assert comp_result[0] == True

        # prior
        r_result2 = r_normal(init = 2.5, mean = 0.1, sd = 2.8)
        prior_obj = Prior(init=[2.5], mean=[0.1], sd=[2.8], distribution='normal')._toR()
        comp_result = base.all_equal(prior_obj, r_result2)
        assert comp_result[0] == True

################### Test Uniform ##################################
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

    def test_uniform_with_r(self):
        # # prior_list
        # r_result1 = r_uniform(init = ro.r('c(0, 0.2)'), min = ro.r('c(-1.0, 0.001)'), max = ro.r('c(1.0, 1.2)'))
        # py_obj1 = [[element for element in sublist] for sublist in r_result1]
        # prior_obj = Prior(init=[0, 0.2], min_val=[-1.0, 0.001], max_val=[1.0, 1.2], distribution='uniform')._toR()
        # for idx in range(len(prior_obj)):
        #     for x, y in zip(prior_obj[idx], py_obj1[idx]):
        #         assert x == y[0]
        #
        # # prior
        # r_result2 = r_uniform(init = 0.2, min = -1.0, max = 1.2)
        # py_obj2 = [[element for element in sublist] for sublist in r_result2]
        # prior_obj = Prior(init=[0.2], min_val=[-1.0], max_val=[1.2], distribution='uniform')._toR()
        # for x, y in zip(prior_obj, py_obj2):
        #     assert x == y[0]

        r_result1 = r_uniform(init = ro.r('c(0, 0.2)'), min = ro.r('c(-1.0, 0.001)'), max = ro.r('c(1.0, 1.2)'))
        prior_obj = Prior(init=[0, 0.2], min_val=[-1.0, 0.001], max_val=[1.0, 1.2], distribution='uniform')._toR()
        comp_result = base.all_equal(prior_obj, r_result1)
        assert comp_result[0] == True

        # prior
        r_result2 = r_uniform(init = 0.2, min = -1.0, max = 1.2)
        prior_obj = Prior(init=[0.2], min_val=[-1.0], max_val=[1.2], distribution='uniform')._toR()
        comp_result = base.all_equal(prior_obj, r_result2)
        assert comp_result[0] == True

################### Test Halfnormal ##################################
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

    def test_halfnormal_with_r(self):
        # # prior_list
        # r_result1 = r_halfnormal(init = ro.r('c(0, 0.2)'), sd = ro.r('c(1.0, 1.2)'))
        # py_obj1 = [[element for element in sublist] for sublist in r_result1]
        # prior_obj = Prior(init=[0, 0.2], sd=[1.0, 1.2], distribution='halfnormal')._toR()
        # for idx in range(len(prior_obj)):
        #     for x, y in zip(prior_obj[idx], py_obj1[idx]):
        #         assert x == y[0]
        #
        # # prior
        # r_result2 = r_halfnormal(init = 0.2, sd = 1.2)
        # py_obj2 = [[element for element in sublist] for sublist in r_result2]
        # prior_obj = Prior(init=[0.2], sd=[1.2],distribution='halfnormal')._toR()
        # for x, y in zip(prior_obj, py_obj2):
        #     assert x == y[0]

        # prior_list
        # Because of different numerical dtype, the integer of Python to R is different
        r_result1 = r_halfnormal(init = ro.r('c(0, 0.2)'), sd = ro.r('c(1.0, 1.2)'))
        prior_obj = Prior(init=[0., 0.2], sd=[1.0, 1.2], distribution='halfnormal')._toR()
        comp_result = base.all_equal(prior_obj, r_result1)
        assert comp_result[0] == True

        # prior
        r_result2 = r_halfnormal(init = 0.2, sd = 1.2)
        prior_obj = Prior(init=[0.2], sd=[1.2],distribution='halfnormal')._toR()
        comp_result = base.all_equal(prior_obj, r_result2)
        assert comp_result[0] == True

################### Test Tnormal ##################################
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

    def test_tnormal_with_r(self):
        # # prior_list
        # r_result1 = r_tnormal(init = ro.r('c(2, 2.2)'), mean = ro.r('c(-1.0, 0.001)'),
        #                       min = ro.r('c(1.2, 2)'), max = 3.3,
        #                       sd = ro.r('c(1.0, 1.2)'))
        # py_obj1 = [[element for element in sublist] for sublist in r_result1]
        # prior_obj = Prior(init=[2, 2.2], mean=[-1.0, 0.001], min_val=[1.2, 2],
        #                   max_val=[3.3], sd=[1.0, 1.2], distribution='tnormal')._toR()
        # for idx in range(len(prior_obj)):
        #     for x, y in zip(prior_obj[idx], py_obj1[idx]):
        #         assert x == y[0]
        #
        # # prior
        # r_result2 = r_tnormal(init = 2.2, mean= -1.0, min = 2, max = 3.3, sd=1.2)
        # py_obj2 = [[element for element in sublist] for sublist in r_result2]
        # prior_obj = Prior(init=[2.2], mean=[-1.0], min_val=[2], max_val=[3.3], sd=[1.2],
        #                   distribution='tnormal')._toR()
        # for x, y in zip(prior_obj, py_obj2):
        #     assert x == y[0]
        #
        # # prior - np.inf
        # r_result2 = r_tnormal(init = 2.2, mean= -1.0, max = 3.3, sd=1.2)
        # py_obj2 = [[element for element in sublist] for sublist in r_result2]
        # prior_obj = Prior(init=[2.2], mean=[-1.0], max_val=[3.3], sd=[1.2],
        #                   distribution='tnormal')._toR()
        # for x, y in zip(prior_obj, py_obj2):
        #     assert x == y[0]

        # prior_list
        r_result1 = r_tnormal(init=ro.r('c(2, 2.2)'), mean=ro.r('c(-1.0, 0.001)'),
                              min=ro.r('c(1.2, 2)'), max=3.3,
                              sd=ro.r('c(1.0, 1.2)'))
        prior_obj = Prior(init=[2, 2.2], mean=[-1.0, 0.001], min_val=[1.2, 2],
                          max_val=[3.3], sd=[1.0, 1.2], distribution='tnormal')._toR()
        comp_result = base.all_equal(prior_obj, r_result1)
        assert comp_result[0] == True

        # prior
        r_result2 = r_tnormal(init=2.2, mean=-1.0, min=2, max=3.3, sd=1.2)
        prior_obj = Prior(init=[2.2], mean=[-1.0], min_val=[2], max_val=[3.3], sd=[1.2],
                          distribution='tnormal')._toR()
        comp_result = base.all_equal(prior_obj, r_result2)
        assert comp_result[0] == True

        # prior - np.inf
        r_result2 = r_tnormal(init=2.2, mean=-1.0, max=3.3, sd=1.2)
        prior_obj = Prior(init=[2.2], mean=[-1.0], max_val=[3.3], sd=[1.2],
                          distribution='tnormal')._toR()
        comp_result = base.all_equal(prior_obj, r_result2)
        assert comp_result[0] == True

################### Test Gamma ##################################
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

    def test_gamma_with_r(self):
        # # prior_list
        # r_result1 = r_gamma(init = ro.r('c(0.1, 0.2)'),
        #                     shape = ro.r('c(1.2, 2)'), rate = ro.r('c(3.0, 3.3)'))
        # py_obj1 = [[element for element in sublist] for sublist in r_result1]
        # prior_obj = Prior(init=[0.1, 0.2], shape=[1.2, 2],
        #                   rate=[3.0, 3.3], distribution='gamma')._toR()
        # for idx in range(len(prior_obj)):
        #     for x, y in zip(prior_obj[idx], py_obj1[idx]):
        #         assert x == y[0]
        #
        # # prior
        # r_result2 = r_gamma(init = 0.2, shape = 1.2, rate = 3.3)
        # py_obj2 = [[element for element in sublist] for sublist in r_result2]
        # prior_obj = Prior(init=[0.2], shape=[1.2], rate=[3.3], distribution='gamma')._toR()
        # for x, y in zip(prior_obj, py_obj2):
        #     assert x == y[0]

        # prior_list
        r_result1 = r_gamma(init = ro.r('c(0.1, 0.2)'),
                            shape = ro.r('c(1.2, 2)'), rate = ro.r('c(3.0, 3.3)'))
        prior_obj = Prior(init=[0.1, 0.2], shape=[1.2, 2],
                          rate=[3.0, 3.3], distribution='gamma')._toR()
        comp_result = base.all_equal(prior_obj, r_result1)
        assert comp_result[0] == True

        # prior
        r_result2 = r_gamma(init = 0.2, shape = 1.2, rate = 3.3)
        prior_obj = Prior(init=[0.2], shape=[1.2], rate=[3.3], distribution='gamma')._toR()
        comp_result = base.all_equal(prior_obj, r_result2)
        assert comp_result[0] == True


