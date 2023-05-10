import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import collections


"""
Implement the Kalman filter via Tensorflow Probability official model

ssm_ulg & ssm_mlg: tfd.LinearGaussianStateSpaceModel
bsm_lg: tfp.sts.LocalLevelStateSpaceModel 

"""


class KalmanFilter:
    """
    Runs the Kalman filter for the given model,
    and returns the filteredx estimates and one-step-ahead predictions of the
    states \eqn{\alpha_t} given the data up to time \eqn{t}. For non-Gaussian models,
    the filtering is based on the approximate Gaussian model.

    Only for model of class lineargaussian, nongaussian or ssm_nlg.


    """
    def __init__(self, **kwargs):

        for key, value in kwargs.items():
            setattr(self, key, value)

        def_kfmethod = getattr(self, f"kf_{kwargs['model_type']}")

        self.infer_result = def_kfmethod(model=kwargs['model'],
                                         model_type_case=kwargs['model_type_case'])

    def kf_linear_gaussian(self, model, model_type_case):
        """Kalman Filter for Linear Gaussian case.
        Args:
            model: bssm model object
            model_type_case: model type case
        Returns:
            infer_result: log-likelihood (approximate in non-Gaussian case),
            one-step-ahead predictions \code{at} and filtered estimates
            \code{att} of states, and the corresponding variances \code{Pt}
            and \code{Ptt} up to the time point n+1 where n is the length of
            the input time series.

        """
