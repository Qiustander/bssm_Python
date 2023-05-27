import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import dtype_util
tfd = tfp.distributions
from tensorflow_probability.python.internal import prefer_static as ps
from runtime_wrap import get_runtime
import numpy as np


class NonlinearSSM(tfd.LinearGaussianStateSpaceModel):
    """Build a linear Gaussian state space model
    Args:
      num_timesteps: Integer Tensor total number of timesteps.
      state_dim: dimension of state variable
      nonlinear_type: nonlinear SSM type denoted as string, currently support type:
      "nlg_linear_gaussian", "nlg_sin_exp", "nlg_growth", "nlg_ar_exp", "sde_poisson_ou", "sde_gbm"
      observation_size: dimension of each observation (>1 for multivariate).
      latent_size: dimension of that state variable.
      initial_state_prior: Optional instance of `tfd.MultivariateNormal`
        representing a prior distribution on the latent state at time
        `initial_step`. must have event shape [latent_size].
      initial_step: Optional scalar `int` `Tensor` specifying the starting
        timestep.
        Default value: 0.
      validate_args: Python `bool`. Whether to validate input
        with asserts. If `validate_args` is `False`, and the inputs are
        invalid, correct behavior is not guaranteed.
        Default value: `False`.
      name: Python `str` name prefixed to ops created by this class.
        Default value: "NonlinearSSM".
    Raises:
      ValueError: if components have different `num_timesteps`.
    """

    # TODO: use decorator to wrap the time-varying function
    def __init__(self,
                 num_timesteps,
                 state_dim,
                 observation_fn,
                 observation_plusnoise_fn,
                 transition_fn,
                 transition_plusnoise_fn,
                 observation_fn_grad,
                 transition_fn_grad,
                 transition_noise_fn,
                 observation_noise_fn,
                 initial_state_prior,
                 initial_step=0,
                 mask=None,
                 experimental_parallelize=False,
                 validate_args=False,
                 allow_nan_stats=True,
                 name='NonlinearSSM'):

        parameters = dict(locals())
        with tf.name_scope(name or 'NonlinearSSM') as name:

            self._num_timesteps = ps.convert_to_shape_tensor(
                num_timesteps, name='num_timesteps')
            self._state_dim = ps.convert_to_shape_tensor(
                state_dim, name='state_dim')
            self._initial_state_prior = initial_state_prior
            self._initial_step = ps.convert_to_shape_tensor(
                initial_step, name='initial_step')

            self._observation_fn_grad = observation_fn_grad
            self._transition_fn_grad = transition_fn_grad
            self._observation_fn = observation_fn
            self._observation_plusnoise_fn = observation_plusnoise_fn
            self._transition_fn = transition_fn
            self._transition_plusnoise_fn = transition_plusnoise_fn
            self._transition_noise_fn = transition_noise_fn
            self._observation_noise_fn = observation_noise_fn

            dtype_list = [initial_state_prior,
                          observation_fn,
                          transition_fn,
                          transition_plusnoise_fn,
                          observation_plusnoise_fn,
                          transition_noise_fn,
                          observation_noise_fn,
                          observation_fn_grad,
                          transition_fn_grad]

            # Infer dtype from time invariant objects. This list will be non-empty
            # since it will always include `initial_state_prior`.
            dtype = dtype_util.common_dtype(
                list(filter(lambda x: not callable(x), dtype_list)),
                dtype_hint=tf.float32)

    @property
    def observation_fn_grad(self):
        return self._observation_fn_grad

    @property
    def transition_fn_grad(self):
        return self._transition_fn_grad

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def state_dim(self):
        return self._state_dim

    @property
    def transition_fn(self):
        return self._transition_fn

    @property
    def transition_noise_fn(self):
        return self._transition_noise_fn

    @property
    def observation_fn(self):
        return self._observation_fn

    @property
    def observation_plusnoise_fn(self):
        return self._observation_plusnoise_fn

    @property
    def transition_plusnoise_fn(self):
        return self._transition_plusnoise_fn

    @property
    def observation_noise_fn(self):
        return self._observation_noise_fn

    @property
    def initial_state_prior(self):
        return self._initial_state_prior


    @classmethod
    def create_model(cls,
                 num_timesteps,
                 observation_size,
                 latent_size,
                 initial_state_mean,
                 initial_state_cov,
                 state_noise_std,
                 obs_noise_std,
                 nonlinear_type,
                rho_state=None,
                mu_state=None,
                input_state=None,
                input_obs=None,
                dtype=tf.float32):
        return cls(**nonlinear_fucntion(
                    function_type=nonlinear_type,
                    obs_len=num_timesteps,
                    obs_dim=observation_size,
                    state_dim=latent_size,
                    obs_noise=obs_noise_std,
                    state_noise=state_noise_std,
                    prior_mean=initial_state_mean,
                    prior_cov=initial_state_cov,
                    input_state=input_state,
                    input_obs=input_obs,
                    rho_state=rho_state,
                    mu_state=mu_state,
                    dtype=dtype
                            ))

def _check_parameters(obs_len,
                       state_dim,
                       obs_dim,
                       state_noise,
                       obs_noise,
                       prior_mean,
                       prior_cov,
                       input_state,
                       input_obs,
                       rho_state,
                       mu_state,
                       dtype,
                       **kwargs):

def _process_tv(model, attritube):
    """
    Args:
        model: model object for time-varying process
        attritbue (str): matrix name of the model object: state_mtx/obs_mtx/state_mtx_noise/obs_mtx_noise
    Returns:
        matrix_tv: callable function that for t-th time point with wrapped matrix
    """

    def matrix_tv(t):
        return tf.linalg.LinearOperatorFullMatrix(tf.gather(tf.convert_to_tensor(getattr(model, attritube))
                                                            , indices=t, axis=-1))

    return matrix_tv

