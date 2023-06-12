import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import dtype_util
tfd = tfp.distributions
from tensorflow_probability.python.internal import prefer_static as ps
import scipy
from runtime_wrap import get_runtime
import numpy as np
from .check_argument import *


class LinearGaussianSSM(tfd.LinearGaussianStateSpaceModel):
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
                 observation_matrix,
                 transition_matrix,
                 observation_noise,
                 transition_noise,
                 initial_state_prior,
                 initial_step=0,
                 mask=None,
                 experimental_parallelize=False,
                 validate_args=False,
                 allow_nan_stats=True,
                 dtype=tf.float32,
                 name='NonlinearSSM',
                 **linear_gaussian_ssm_kwargs):

        parameters = dict(locals())
        parameters.update(linear_gaussian_ssm_kwargs)
        with tf.name_scope(name or 'LinearGaussianSSM') as name:

            self._dtype = dtype
            self._state_dim = ps.convert_to_shape_tensor(
                state_dim, name='state_dim')

            super(LinearGaussianSSM, self).__init__(
                num_timesteps=num_timesteps,
                transition_matrix=transition_matrix,
                transition_noise=transition_noise,
                observation_matrix=observation_matrix,
                observation_noise=observation_noise,
                initial_state_prior=initial_state_prior,
                name=name,
                **linear_gaussian_ssm_kwargs)
            self._parameters = parameters

    @property
    def state_dim(self):
        return self._state_dim

    @property
    def dtype(self):
        return self._dtype

    @classmethod
    def create_model(cls,
                 num_timesteps,
                 observation_size,
                 latent_size,
                 initial_state_mean,
                 state_mtx,
                 obs_mtx,
                 initial_state_cov,
                 state_noise_std,
                 obs_noise_std,
                input_state=None,
                input_obs=None,
                dtype=tf.float32,
                     **kwargs):
        return cls(**_check_parameters(
                    obs_len=num_timesteps,
                    obs_dim=observation_size,
                    state_dim=latent_size,
                    obs_noise=obs_noise_std,
                    state_noise=state_noise_std,
                    state_mtx=state_mtx,
                    obs_mtx=obs_mtx,
                    prior_mean=initial_state_mean,
                    prior_cov=initial_state_cov,
                    input_state=input_state,
                    input_obs=input_obs,
                    dtype=dtype,
                     **kwargs))

def _check_parameters(obs_len,
                       state_dim,
                       obs_dim,
                       obs_mtx,
                       state_mtx,
                       state_noise,
                       obs_noise,
                       prior_mean,
                       prior_cov,
                       input_state,
                       input_obs,
                       dtype,
                        **kwargs):
    """
    Create parameters for linear Gaussian model.
    #TODO: how to create drop-rank std matrix for multivariate normal
    Args:
        obs_len:
        state_dim:
        obs_dim:
        obs_mtx:
        state_mtx:
        state_noise:
        obs_noise:
        prior_mean:
        prior_cov:
        input_state:
        input_obs:
        dtype:
        **kwargs:

    Returns:

    """

    # create Z - obs matrix
    obs_mtx = tf.convert_to_tensor(check_obs_mtx(obs_mtx, obs_dim, obs_len, state_dim), dtype=dtype)
    observation_matrix = obs_mtx \
                    if len(obs_mtx.shape) == 2 else _process_mtx_tv(obs_mtx, dtype)

    # create T - state matrix
    state_mtx = tf.convert_to_tensor(check_state_mtx(state_mtx, state_dim, obs_len), dtype=dtype)
    transition_matrix = state_mtx\
                    if len(state_mtx.shape) == 2 else _process_mtx_tv(state_mtx, dtype)

    # create R - noise state matrix
    state_mtx_noise = tf.convert_to_tensor(check_state_noise(state_noise, state_dim, obs_len), dtype=dtype) # already 3 dim
    input_state = tf.convert_to_tensor(check_input_state(input_state, state_dim, obs_len), dtype=dtype)
    time_vary_state_noise = False
    if input_state.shape[-1] == obs_len or state_mtx_noise.shape[-1] == obs_len:
        input_state = tf.repeat(input_state[..., None], obs_len, axis=-1) \
                if input_state.shape[-1] != obs_len else input_state
        state_mtx_noise = tf.repeat(state_mtx_noise[..., None], obs_len, axis=-1) \
                if state_mtx_noise.shape[-1] != obs_len else state_mtx_noise
        time_vary_state_noise = True
    # state_mtx_noise_cov = tf.einsum('ijk,ljk->ilk', state_mtx_noise, state_mtx_noise) if time_vary_state_noise\
    #                     else tf.einsum('ij,lj->il', state_mtx_noise, state_mtx_noise)


    # create H - noise obs matrix
    time_vary_obs_noise = False
    obs_mtx_noise = tf.convert_to_tensor(check_obs_mtx_noise(obs_noise, obs_dim, obs_len), dtype=dtype)
    input_obs = tf.convert_to_tensor(check_input_obs(input_obs, obs_dim, obs_len), dtype=dtype)
    if input_obs.shape[-1] == obs_len or obs_mtx_noise.shape[-1] == obs_len:
        input_obs = tf.repeat(input_obs[..., None], obs_len, axis=-1) \
                if input_obs.shape[-1] != obs_len else input_obs
        obs_mtx_noise = tf.repeat(obs_mtx_noise[..., None], obs_len, axis=-1) \
                if obs_mtx_noise.shape[-1] != obs_len else obs_mtx_noise
        time_vary_obs_noise = True
    # obs_mtx_noise_cov = tf.einsum('ijk,ljk->ilk', obs_mtx_noise, obs_mtx_noise) if time_vary_obs_noise\
    #                     else tf.einsum('ij,lj->il', obs_mtx_noise, obs_mtx_noise)

    observation_noise = tfd.MultivariateNormalLinearOperator(
                                 loc=input_obs,
                            scale=tf.linalg.LinearOperatorFullMatrix(obs_mtx_noise)) \
                if not time_vary_obs_noise else\
        lambda x: tfd.MultivariateNormalLinearOperator(
                                 loc=input_obs[..., x],
                            scale=tf.linalg.LinearOperatorFullMatrix(obs_mtx_noise[..., x]))
    transition_noise = tfd.MultivariateNormalLinearOperator(
                                 loc=input_state,
                            scale=tf.linalg.LinearOperatorFullMatrix(state_mtx_noise)) \
                if not time_vary_state_noise else\
        lambda x: tfd.MultivariateNormalLinearOperator(
                                 loc=input_state[..., x],
                            scale=tf.linalg.LinearOperatorFullMatrix(state_mtx_noise[..., x]))

    prior_mean = tf.convert_to_tensor(check_prior_mean(prior_mean, state_dim), dtype=dtype)
    prior_cov = tf.convert_to_tensor(check_prior_cov(prior_cov, state_dim), dtype=dtype)

    initial_state_prior = tfd.MultivariateNormalFullCovariance(
        loc=prior_mean,
        covariance_matrix=prior_cov)

    return {"observation_matrix":observation_matrix, "transition_matrix":transition_matrix,
               "transition_noise":transition_noise, "observation_noise":observation_noise,
                    "initial_state_prior":initial_state_prior, "num_timesteps":obs_len, "state_dim": state_dim}


def _process_mtx_tv(time_vary_mtx, dtype=tf.float32):
    """
    Args:
        time_vary_mtx: matrix name of the model object: state_mtx/obs_mtx/state_mtx_noise/obs_mtx_noise
        dtype: data type of the matrix
    Returns:
        matrix_tv: callable function that for t-th time point with wrapped matrix
    """

    def matrix_tv(t):
        return tf.linalg.LinearOperatorFullMatrix(tf.gather(tf.convert_to_tensor(time_vary_mtx, dtype=dtype)
                                                            , indices=t, axis=-1))

    return matrix_tv

