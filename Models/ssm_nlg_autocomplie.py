import tensorflow as tf
import tensorflow_probability as tfp
from .nonlinear_function_type import nonlinear_fucntion
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import dtype_util
tfd = tfp.distributions
from tensorflow_probability.python.internal import prefer_static as ps
from functools import partial


class NonlinearSSM(distribution.AutoCompositeTensorDistribution):
    """Build a state space model with nonlinear state and observation functions and Gaussian noise.
    Args:
      nonlinear_type: nonlinear SSM type denoted as string, currently support type:
      "nlg_linear_gaussian", "nlg_sin_exp", "nlg_growth", "nlg_ar_exp", "sde_poisson_ou", "sde_gbm"
      num_timesteps: Integer Tensor total number of timesteps.
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
      **linear_gaussian_ssm_kwargs: Optional additional keyword arguments to
        to the base `tfd.LinearGaussianStateSpaceModel` constructor.
    Raises:
      ValueError: if components have different `num_timesteps`.
    """
    # TODO: AutoCompositeTensorDistribution is not compatible with @tf.cuntion because the input are callbel. Why Linear GAUSSIAN can ? Because they input callable in forward filter
    # TODO: use decorator to wrap the time-varying function
    def __init__(self,
                 num_timesteps,
                 observation_fn,
                 transition_fn,
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
            self._initial_state_prior = initial_state_prior
            self._initial_step = ps.convert_to_shape_tensor(
                initial_step, name='initial_step')

            self._observation_fn_grad = observation_fn_grad
            self._transition_fn_grad = transition_fn_grad
            self._observation_fn = observation_fn
            self._transition_fn = transition_fn
            self._transition_noise_fn = transition_noise_fn
            self._observation_noise_fn = observation_noise_fn
            # self._initial_state_prior = initial_state_prior

            dtype_list = [initial_state_prior,
                          observation_fn,
                          transition_fn,
                          transition_noise_fn,
                          observation_noise_fn,
                          observation_fn_grad,
                          transition_fn_grad]

            # Infer dtype from time invariant objects. This list will be non-empty
            # since it will always include `initial_state_prior`.
            dtype = dtype_util.common_dtype(
                list(filter(lambda x: not callable(x), dtype_list)),
                dtype_hint=tf.float32)

        super(NonlinearSSM, self).__init__(
            dtype=dtype,
            reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters=parameters,
            name=name)

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
    def transition_fn(self):
        return self._transition_fn

    @property
    def transition_noise_fn(self):
        return self._transition_noise_fn

    @property
    def observation_fn(self):
        return self._observation_fn

    @property
    def observation_noise_fn(self):
        return self._observation_noise_fn

    @property
    def initial_state_prior(self):
        return self._initial_state_prior


    @classmethod
    def create_dist(cls,
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

    #TODO: rewrite the EKF -> IEKF
    @partial(tf.function, reduce_retracing=True, autograph=False)
    def extended_Kalman_filter(self, observations):
        """ Conduct the extended Kalman Filter
        Args:
            observations: observed time series
        Returns:
            infer_result of EKF
        """
        return tfp.experimental.sequential.extended_kalman_filter(
            observations,
            self.initial_state_prior,
            self.transition_fn,
            self.observation_fn,
            self.transition_fn_grad,
            self.observation_fn_grad,
            name=None
        )
