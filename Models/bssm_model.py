import numpy as np
from check_argument import CheckArg

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
        self.check_arg = CheckArg() # initialize the checking class
        self.state_dim = input_dict['state_dim']

    def ssm_ulg(self, y):
        """
        Args:
            y (np.array): Observations as time series (or vector) of length \eqn{n}.
            input_dict:
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
            noise_std (np.array): A vector of standard deviations. Either a scalar or a vector of  length n.
            state_names (str): A character vector defining the names of the states.
        return:
        """

        if 'state_names' not in self.input_dict:
            state_names = [f"state {i+1}" for i in range(m)]
            self.input_dict['state_names'] = state_names
        if 'init_theta' not in self.input_dict:
            init_theta = np.array([])
            self.input_dict['init_theta'] = init_theta

        #check


    def ssm_mlg(self):
        pass

    def ssm_ung(self):
        pass

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
        pass

    def ar1_lg(self):
        pass

    def ar1_ng(self):
        pass


