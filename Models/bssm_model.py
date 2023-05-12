import numpy as np
from .check_argument import *
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
import pandas as pd
numpy2ri.activate()
pandas2ri.activate()
base = importr('base', lib_loc="/usr/lib/R/library")
stat = importr('stats', lib_loc="/usr/lib/R/library")
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


# TODO: Combine the check of all models into one function
# TODO: Rewrite all SSMModel as the TFP class

class SSModel:
    """
    State space model class for various kinds of state space models, including:
    ssm_ulg: General univariate linear-Gaussian state space models
    ssm_mlg: General multivariate linear-Gaussian state space models
    ssm_ung General univariate Non-Gaussian state space model
    ssm_mng: General multivariate Non-Gaussian State Space Model
    ssm_nlg: General multivariate nonlinear Gaussian state space models
    ssm_sde: Univariate state space model with continuous SDE dynamics
    ssm_svm: Stochastic Volatility Model in state space model
    bsm_lg: Gaussian Basic Structural (Time Series) Model
    bsm_ng: Non-Gaussian Basic Structural (Time Series) Model
    ar1_lg: Univariate Gaussian model with AR(1) latent process
    ar1_ng: Non-Gaussian model with AR(1) latent process
    """

    def __init__(self,  **kwargs):
        """
        Args:
            kwargs: definition of input_arguments for different models
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

        def_model = getattr(self, f"{kwargs['model_name']}")
        def_model()

    def ssm_ulg(self):
        """General univariate linear-Gaussian state space models
        Args:
            y (np.array): Observations as time series (or vector) of length \eqn{n}.
            obs_mtx (np.array): System matrix Z of the observation equation. Either a vector of length m, a m x n matrix.
            state_mtx (np.array): System matrix T of the state equation. Either a m x m matrix or a m x m x n array.
            state_mtx_noise (np.array): Noise coefficient matrix R (lower triangular) of the state equation. Either a m x k matrix or a m x k x n array.
            prior_mean (np.array): Prior mean for the initial state as a vector of length m.
            prior_cov (np.array): Prior covariance matrix for the initial state as m x m matrix.
            input_state (np.array): Intercept terms \eqn{C_t} for the state equation, given as a m times 1 or m times n matrix.
            input_obs (np.array): Intercept terms \eqn{D_t} for the observations equation, given as a scalar or vector of length n.
            init_theta (np.array): Initial values for the unknown hyperparameters theta (i.e. unknown variables excluding latent state variables).
            obs_mtx_noise (np.array): Noise coefficient matrix H (lower triangular) for the observed euqation.
                                        Denoted as standard deviations. Either a scalar or a vector of length n.
        Returns:
        """
        m = self.state_dim
        self.y = self.y[..., None].astype("float32")
        size_y = check_y(self.y) # return time length and feature numbers
        n = size_y[0]

        # create Z - obs matrix
        self.obs_mtx = check_obs_mtx(self.obs_mtx, 1, n, m).astype("float32")

        # create T - state matrix
        self.state_mtx = check_state_mtx(self.state_mtx, m, n).astype("float32")

        # create R - noise state matrix
        self.state_mtx_noise = check_state_noise(self.state_mtx_noise, m, n).astype("float32")

        self.prior_mean = check_prior_mean(self.prior_mean, m).astype("float32")
        self.prior_cov = check_prior_cov(self.prior_cov, m).astype("float32")

        self.input_obs = check_input_obs(self.input_obs, 1, n).astype("float32")
        self.input_state = check_input_state(self.input_state, m, n).astype("float32")

        self.obs_mtx_noise = check_obs_mtx_noise(self.obs_mtx_noise, 1, n, multivariate=False).astype("float32")
        self.model_type = "linear_gaussian"

    def ssm_mlg(self):
        """General multivariate linear Gaussian state space models
        Args:
            y (np.array): Observations as multivariate time series as matrix with dimension n x p.
            obs_mtx (np.array): System matrix Z of the observation equation. Either p x m matrix or p x m x n tensor.
            state_mtx (np.array): System matrix T of the state equation. Either a m x m matrix or a m x m x n tensor.
            state_mtx_noise (np.array): Lower triangular matrix R the state equation. Either a m x k matrix or a m x k x n array.
            input_state (np.array): Intercept terms \eqn{C_t} for the state equation, given as m x n matrix.
            input_obs (np.array): Intercept terms \eqn{D_t} for the observations equation, given as a p x n matrix.
            obs_mtx_noise (np.array):Noise coefficient matrix H (lower triangular) for the observed euqation.
                                        Denoted as standard deviations. Either a scalar or a vector of length n.
            Others refer to ssm_ulg.
        Returns:
        """
        m = self.state_dim
        size_y = check_y(self.y, multivariate=True) # return time length and feature numbers
        n, p = size_y

        # create Z - obs matrix
        self.obs_mtx = check_obs_mtx(self.obs_mtx, p, n, m, multivariate=True)

        # create T - state matrix
        self.state_mtx = check_state_mtx(self.state_mtx, m, n)

        # create R - noise state matrix
        self.state_mtx_noise = check_state_noise(self.state_mtx_noise, m, n)

        self.prior_mean = check_prior_mean(self.prior_mean, m)
        self.prior_cov = check_prior_cov(self.prior_cov, m)

        self.input_obs = check_input_obs(self.input_obs, p, n)
        self.input_state = check_input_state(self.input_state, m, n)

        self.obs_mtx_noise = check_obs_mtx_noise(self.obs_mtx_noise, p, n, multivariate=True)

        self.model_type = "linear_gaussian"

    def ssm_ung(self):
        """
        General univariate non-Gaussian state space model
        Args:
            input_dict:
            y (np.array): Observations as time series (or vector) of length \eqn{n}.
            obs_mtx (np.array): System matrix Z of the observation equation. Either a vector of length m, a m x n matrix.
            state_mtx (np.array): System matrix T of the state equation. Either a m x m matrix or a m x m x n array.
            state_mtx_noise (np.array): Lower triangular matrix R the state equation. Either a m x k matrix or a m x k x n array.
            prior_mean (np.array): Prior mean for the initial state as a vector of length m.
            prior_cov (np.array): Prior covariance matrix for the initial state as m x m matrix.
            input_state (np.array): Intercept terms \eqn{C_t} for the state equation, given as a m times 1 or m times n matrix.
            input_obs (np.array): Intercept terms \eqn{D_t} for the observations equation, given as a scalar or vector of length n.
            init_theta (np.array): Initial values for the unknown hyperparameters theta (i.e. unknown variables excluding latent state variables).
            update_fn (object): A function which returns list of updated model  components given input vector theta.
                    This function should take only one  vector argument which is used to create list with elements named as Z, T, R, a1, P1, D, C, and phi,
                    where each element matches the dimensions of the original model. If any of these components is missing, it is assumed to be constant wrt.
                     theta. It's best to check the internal dimensions with str(model_object) as the dimensions of input arguments can differ from the final dimensions.
                     If any of these components is missing, it is assumed to be constant wrt. the eta.
            prior_fn (object): A function which returns log of prior density given input vector theta.
            obs_mtx_noise (np.array): A vector H of standard deviations. Either a scalar or a vector of  length n.
        Returns:

        """
        self.model_type = "non_gaussian"


    def ssm_mng(self):
        self.model_type = "non_gaussian"

    def ssm_nlg(self):
        pass
    def ssm_sde(self):
        pass
    def ssm_svm(self):
        self.model_type = "non_gaussian"


    def bsm_lg(self):
        self.model_type = "linear_gaussian"

    def bsm_ng(self):
        """Non-Gaussian Basic Structural (Time Series) Model
            Constructs a non-Gaussian basic structural model with local level or
            local trend component, a seasonal component, and regression component
            (or subset of these components).
        Args:
            input_dict:
            y (np.array): Observations as a vector of length \eqn{n}.
            sd_level (np.array): Standard deviation of the noise of level equation.
                Should be a vecotr of prior function or a scalar value defining a known value such as 0.
            sd_slope (np.array): Standard deviation of the noise of slope equation.
                Should be a vecotr of prior function or a scalar value defining a known value such as 0, or missing, in which case the slope term is omitted from the model.
            sd_seasonal (np.array): Standard deviation of the noise of seasonal equation. Should be a vecotr of prior function, scalar value defining a known value such as 0,
                or missing, in which case the seasonal term is omitted from the model.
            sd_noise (np.array):  prior for the standard deviation of the additional noise term to be added to linear predictor,
                defined as a vector of prior. If missing, no additional noise term is used.
            phi: Additional parameter relating to the non-Gaussian distribution.
                For negative binomial distribution this is the dispersion term,
                for gamma distribution this is the shape parameter, and for other distributions this is ignored.
                Should a vector of prior function or a positive scalar.
            positive_const: A vector u of positive constants for non-Gaussian models.
                For Poisson, gamma, and negative binomial distribution, this corresponds to the offset term. For binomial, this is the number of trials.
            beta: A prior for the regression coefficients. Should be a vector of prior function (in case of multiple coefficients) or missing in case of no covariates.
            xreg: A matrix containing covariates with number of rows matching the length of observation y.
            prior_mean (np.array): Prior means for the initial states (level, slope, seasonals). Defaults to vector of zeros.
            prior_cov (np.array): Prior covariance matrix for the initial states (level, slope, seasonals). Default is diagonal matrix with 100 on the diagonal.
            input_state (np.array): Intercept terms for state equation, given as a m x n or m x 1 matrix.
            init_theta (np.array): Initial values for the unknown hyperparameters theta (i.e. unknown variables excluding latent state variables).
            period (int): Length of the seasonal pattern. Must be a positive value greater than 2
                and less than the length of the input time series. Default is frequency(y), which can also return non-integer value (in which case error is given).
            distribution (str): Distribution of the observed time series.
                Possible choices are "poisson", "binomial", "gamma", and "negative binomial".
        Returns:

        """

        self.model_type = "non_gaussian"

    def ar1_lg(self):
        self.model_type = "linear_gaussian"


    def ar1_ng(self):
        self.model_type = "non_gaussian"



    # For R usage

    def model_type_case(self):
        if self.model_type == "linear_gaussian":
            return ["ssm_mlg", "ssm_ulg", "bsm_lg", "ar1_lg"].index(self.model_name)
        else:
            return ["ssm_mng", "ssm_ung", "bsm_ng", "svm", "ar1_ng"].index(self.model_name)

    def update_fn(self):
        """ Define a function which returns list of updated model  components given input vector theta.
        This function should take only one  vector argument which is used to create list with elements named as Z, T, R, a1, P1, D, C, and phi,
        where each element matches the dimensions of the original model. If any of these components is missing, it is assumed to be constant wrt.
         theta. It's best to check the internal dimensions with str(model_object) as the dimensions of input arguments can differ from the final dimensions.
         If any of these components is missing, it is assumed to be constant wrt. the eta.
        Returns:

        """

    def prior_fn(self):
        """Define a function which returns log of prior density
            given input vector theta.
        Returns:

        """

    def _toRssmulg(self, prior_fn, update_fn):
        """Transfer to R object (S3)
        Args:
            prior_fn: prior function, R object
            update_fn: update function, R object

        Returns:
            r_obj: ssm_model for R
        """
        # the prior function and update function, use the original version.

        """
        Some translation from TFP to R:
        obs_mtx: observation_size, latent_size, time_length -> Z univarate only m, or m x n, no observation, multivariate three dim
        state_mtx: latent_size, latent_size, timelength -> T no problem
        input_obs: observation_size -> D  univariate: vector, multivariate: p x n
        input_state: latent_size -> C univaraite m x 1. m x n multivariate  m x n 
        obs_noise: observation_size, observation_size -> H  univaraite: 1 or n, multivaraite: p x px n
        state_noise: latent_size, latent_size -> R no problem
        """
        model_dict = self.__dict__
        if "obs_mtx" in model_dict: model_dict["Z"] = model_dict.pop("obs_mtx")
        if "state_mtx" in model_dict: model_dict["T"] = model_dict.pop("state_mtx")
        if len(model_dict["T"].shape) == 2: #state mtx
            model_dict["T"] = model_dict["T"][..., None]
        if "state_mtx_noise" in model_dict: model_dict["R"] = model_dict.pop("state_mtx_noise")
        if len(model_dict["R"].shape) == 2: #state mtx noise
            model_dict["R"] = model_dict["R"][..., None]
        if "prior_mean" in model_dict: model_dict["a1"] = model_dict.pop("prior_mean")
        if "obs_mtx_noise" in model_dict: model_dict["H"] = model_dict.pop("obs_mtx_noise")
        if "prior_cov" in model_dict: model_dict["P1"] = model_dict.pop("prior_cov")
        if "input_state" in model_dict: model_dict["C"] = model_dict.pop("input_state")
        if len(model_dict["C"].shape) == 1: # obs noise
            model_dict["C"] = model_dict["C"][..., None]
        if "input_obs" in model_dict: model_dict["D"] = model_dict.pop("input_obs")
        model_dict["theta"] = model_dict.pop("init_theta")
        model_type = model_dict.pop("model_type")
        if model_type == "linear_gaussian":
            model_type = "lineargaussian"
        model_name = model_dict.pop("model_name")
        model_dict.pop("state_dim")

        if model_name == "ssm_ulg":
            model_dict["Z"] = model_dict["Z"][0]
            if len(model_dict["H"].shape) == 2:  # obs noise
                model_dict["H"] = model_dict["H"][0]
            model_dict["xreg"] = np.zeros((0, 0))
            model_dict["beta"] = ro.FloatVector([])
            model_dict["y"] = pd.Series(model_dict["y"].flatten())
            name_order = ["y", "Z", "H", "T", "R", "a1", "P1",
                      "D", "C", "update_fn", "prior_fn", "theta", "xreg", "beta"] # corresponding to R
        elif model_name == "ssm_mlg":
            if len(model_dict["Z"].shape) == 2:  # state mtx
                model_dict["Z"] = model_dict["Z"][..., None]
            if len(model_dict["C"].shape) == 1:  # obs noise
                model_dict["C"] = model_dict["C"][..., None]
            if len(model_dict["D"].shape) == 1:  # obs noise
                model_dict["D"] = model_dict["D"][..., None]
            if len(model_dict["H"].shape) == 2:  # obs noise
                model_dict["H"] = model_dict["H"][..., None]
            name_order = ["y", "Z", "H", "T", "R", "a1", "P1",
                      "D", "C", "update_fn", "prior_fn", "theta"] # corresponding to R
        model_dict["prior_fn"] = prior_fn
        model_dict["update_fn"] = update_fn


        r_obj = ro.ListVector.from_length(len(model_dict))
        for i, x in enumerate(name_order):
            with localconverter(ro.default_converter + pandas2ri.converter + ro.numpy2ri.converter):
                obj_convert = ro.conversion.py2rpy(model_dict[x])
                r_obj[i] = obj_convert

        r_obj.do_slot_assign("names", ro.StrVector(name_order))
        r_obj.do_slot_assign("class", ro.StrVector([model_name, model_type, "bssm_model"]))

        return r_obj
