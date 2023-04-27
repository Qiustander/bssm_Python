import numpy as np
from check_argument import *

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
    bsm_lg: Basic Structural (Time Series) Model
    bsm_ng: Non-Gaussian Basic Structural (Time Series) Model
    ar1_lg: Univariate Gaussian model with AR(1) latent process
    ar1_ng: Non-Gaussian model with AR(1) latent process
    """

    def __init__(self, input_dict):
        """
        Args:
            input_dict: definition of input_arguments for different models
        """
        self.input_dict = input_dict
        self.state_dim = input_dict['state_dim']

    def ssm_ulg(self):
        """General univariate linear-Gaussian state space models
        Args:
            input_dict:
            y (np.array): Observations as time series (or vector) of length \eqn{n}.
            obs_mtx (np.array): System matrix Z of the observation equation. Either a vector of length m, a m x n matrix.
            state_mtx (np.array): System matrix T of the state equation. Either a m x m matrix or a m x m x n array.
            state_mtx_lower (np.array): Lower triangular matrix R the state equation. Either a m x k matrix or a m x k x n array.
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
            noise_std (np.array): A vector H of standard deviations. Either a scalar or a vector of  length n.
            state_names (str): A character vector defining the names of the states.
        Returns:
        """
        #TODO: first initilize the dictionary with None -> input -> do the checking according to variable name

        size_y = check_y(self.input_dict['y']) # return time length and feature numbers
        n = size_y[0]

        # create Z - obs matrix
        obs_mtx = check_obs_mtx(self.input_dict['obs_mtx'], 1, n)
        m = obs_mtx.shape[0]
        self.input_dict['obs_mtx'] = obs_mtx

        # create T - state matrix
        self.input_dict['state_mtx']  = check_state_mtx(self.input_dict['state_mtx'], m, n)

        # create R - lower state matrix
        self.input_dict['state_mtx_lower'] = check_mtx_lower(self.input_dict['state_mtx_lower'], m, n)

        self.input_dict['prior_mean'] = check_prior_mean(self.input_dict['prior_mean'], m)
        self.input_dict['prior_cov'] = check_prior_cov(self.input_dict['prior_cov'], m)

        self.input_dict['input_obs'] = check_input_obs(self.input_dict['input_obs'], 1, n)
        self.input_dict['input_state'] = check_input_state(self.input_dict['input_state'], m, n)

        self.input_dict['noise_std'] = check_noise_std(self.input_dict['noise_std'], 1, n)

        if 'state_names' not in self.input_dict:
            state_names = [f"state {i+1}" for i in range(m)]
            self.input_dict['state_names'] = state_names
        if 'init_theta' not in self.input_dict:
            init_theta = np.array([])
            self.input_dict['init_theta'] = init_theta

    def ssm_mlg(self):
        """General multivariate linear Gaussian state space models
        Args:
            input_dict:
            y (np.array): Observations as multivariate time series as matrix with dimension n x p.
            obs_mtx (np.array): System matrix Z of the observation equation. Either p x m matrix or p x m x n tensor.
            state_mtx (np.array): System matrix T of the state equation. Either a m x m matrix or a m x m x n tensor.
            state_mtx_lower (np.array): Lower triangular matrix R the state equation. Either a m x k matrix or a m x k x n array.
            input_state (np.array): Intercept terms \eqn{C_t} for the state equation, given as m x n matrix.
            input_obs (np.array): Intercept terms \eqn{D_t} for the observations equation, given as a p x n matrix.
            noise_std (np.array): A vector H of standard deviations. Either a scalar or a vector of  length n.

            Others refer to ssm_ulg.
        Returns:
        """

        size_y = check_y(self.input_dict['y'])  # return time length and feature numbers
        n, p = size_y

        # create Z - obs matrix
        obs_mtx = check_obs_mtx(self.input_dict['obs_mtx'], p, n, multivariate=True)
        m = obs_mtx.shape[-1]
        self.input_dict['obs_mtx'] = obs_mtx

        # create T - state matrix
        self.input_dict['state_mtx']  = check_state_mtx(self.input_dict['state_mtx'], m, n)

        # create R - lower state matrix
        self.input_dict['state_mtx_lower'] = check_mtx_lower(self.input_dict['state_mtx_lower'], m, n)

        self.input_dict['prior_mean'] = check_prior_mean(self.input_dict['prior_mean'], m)
        self.input_dict['prior_cov'] = check_prior_cov(self.input_dict['prior_cov'], m)

        self.input_dict['input_obs'] = check_input_obs(self.input_dict['input_obs'], p, n)
        self.input_dict['input_state'] = check_input_state(self.input_dict['input_state'], m, n)

        self.input_dict['noise_std'] = check_noise_std(self.input_dict['noise_std'], p, n, multivariate=True)

        # if 'state_names' not in self.input_dict:
        #     state_names = [f"state {i+1}" for i in range(m)]
        #     self.input_dict['state_names'] = state_names
        # if self.input_dict['init_theta'] is None:
        #     self.input_dict['init_theta'] = np.array([])

    def ssm_ung(self):
        """
        General univariate non-Gaussian state space model
        Args:
            input_dict:
            y (np.array): Observations as time series (or vector) of length \eqn{n}.
            obs_mtx (np.array): System matrix Z of the observation equation. Either a vector of length m, a m x n matrix.
            state_mtx (np.array): System matrix T of the state equation. Either a m x m matrix or a m x m x n array.
            state_mtx_lower (np.array): Lower triangular matrix R the state equation. Either a m x k matrix or a m x k x n array.
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
            noise_std (np.array): A vector H of standard deviations. Either a scalar or a vector of  length n.
            state_names (str): A character vector defining the names of the states.
        Returns:

        """


    def ssm_mng(self):
        pass

    def ssm_nlg(self):
        pass

    def ssm_sde(self):
        pass

    def ssm_svm(self):
        pass

    def bsm_lg(self):
        pass

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
                For negative binomial distribution this is the dispersion term, for gamma distribution this is the shape parameter, and for other distributions this is ignored.
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
        if self.input_dict['distribution'] not in self.dist_list:
            raise AttributeError("No distribution found. Please check again.")

        size_y = check_y(self.input_dict['y'], multivariate=False, distribution=self.input_dict['distribution'])  # return time length and feature numbers
        n = size_y[0]

        self.input_dict['positive_const'] = check_positive_const(self.input_dict['positive_const'], self.input_dict['y'])

        self.input_dict['state_mtx_lower'] = check_mtx_lower(self.input_dict['state_mtx_lower'], m, n)

        self.input_dict['prior_mean'] = check_prior_mean(self.input_dict['prior_mean'], m)
        self.input_dict['prior_cov'] = check_prior_cov(self.input_dict['prior_cov'], m)

        self.input_dict['input_obs'] = check_input_obs(self.input_dict['input_obs'], p, n)
        self.input_dict['input_state'] = check_input_state(self.input_dict['input_state'], m, n)

        self.input_dict['noise_std'] = check_noise_std(self.input_dict['noise_std'], p, n, multivariate=True)


    def ar1_lg(self):
        pass

    def ar1_ng(self):
        pass


