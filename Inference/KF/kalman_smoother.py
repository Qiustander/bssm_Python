from Models.check_argument import check_missingness
from rpy2.robjects.packages import importr
import os.path as pth
base = importr('base', lib_loc="/usr/lib/R/library")
bssm = importr('bssm', lib_loc=f"{pth.expanduser('~')}/R/x86_64-pc-linux-gnu-library/4.3")


class KalmanSmoother:
    """
    Runs the Kalman smoothing for the given model,
    and returns the filtered estimates and one-step-ahead predictions of the
    states \eqn{\alpha_t} given the data up to time \eqn{t}. For non-Gaussian models,
    the filtering is based on the approximate Gaussian model.

    """
    def __init__(self, **kwargs):

        for key, value in kwargs.items():
            setattr(self, key, value)

        def_kfmethod = getattr(self, f"ksmoother_{kwargs['model_type']}")

        self.infer_result = def_kfmethod(model=kwargs['model'],
                                         model_type_case=kwargs['model_type_case'])

    def ksmoother_linear_gaussian(self, model, model_type_case):
        """Kalman Soother for Linear Gaussian case.
        Args:
            model: bssm model object
            model_type_case: model type case
        Returns:
            infer_result: smoothed estimates of the states, and smoothed variances.

        """
        # check_missingness(model)

        infer_result = bssm.gaussian_smoother(model, model_type=model_type_case)

        return infer_result