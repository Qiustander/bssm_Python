import tensorflow as tf
import tensorflow_probability as tfp
from .nonlinear_function_type import nonlinear_fucntion
from tensorflow_probability.python.internal import dtype_util
tfd = tfp.distributions
from tensorflow_probability.python.internal import prefer_static as ps
from runtime_wrap import get_runtime


class NonlinearSSM(object):
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
    @tf.function
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

    def unscented_Kalman_filter(self, observations):
        pass

    # @get_runtime(loop_time=10)
    def ensemble_Kalman_filter(self, observations, num_particles, dampling=1):
        """Ensemble Kalman filter estimation
        Args:
            observations:
            num_particles: number of ensembles at each step. Could be time-varying
            dampling: Floating-point `Tensor` representing how much to damp the
                update by. Used to mitigate filter divergence. Default value: 1.
        Returns:

      References

          [1] Geir Evensen. Sequential data assimilation with a nonlinear
              quasi-geostrophic model using Monte Carlo methods to forecast error
              statistics. Journal of Geophysical Research, 1994.

          [2] Matthias Katzfuss, Jonathan R. Stroud & Christopher K. Wikle
              Understanding the Ensemble Kalman Filter.
              The Americal Statistician, 2016.

          [3] Jeffrey L. Anderson and Stephen L. Anderson. A Monte Carlo Implementation
              of the Nonlinear Filtering Problem to Produce Ensemble Assimilations and
              Forecasts. Monthly Weather Review, 1999.

        """
        prior_samples = self.initial_state_prior.sample(num_particles)

        @tf.function
        def ekf_one_step(filtered_ensembles, current_observation):
            # TODO: now use event shape = num_particles, perhaps modify the function to batch shape for speed up
            # current_observation = input_all[1]
            # If observations are scalar, we can avoid some matrix ops.
            observation_size_is_static_and_scalar = (current_observation.shape[-1] == 1)

            ############### Estimation
            state_prior_samples = tf.vectorized_map(lambda x:
                                  self.transition_fn(x).sample(), filtered_ensembles)

            ########### Correction
            correct_samples = tf.vectorized_map(lambda x:
                                                    self.observation_fn(x).sample(), state_prior_samples)

            # corrected_mean = tf.reduce_mean(correct_samples)
            corrected_cov = self._covariance(correct_samples)

            # covariance_between_state_and_predicted_observations
            # Cov(X, G(X))  = (X - μ(X))(G(X) - μ(G(X)))ᵀ
            covariance_xy = tf.nest.map_structure(
                lambda x: self._covariance(x, correct_samples),
                state_prior_samples)

            # covariance_predicted_observations
            covriance_yy = tf.linalg.LinearOperatorFullMatrix(
                corrected_cov,
                # SPD because _linop_covariance(observation_particles_dist) is SPD
                # and _covariance(predicted_observation_particles) is SSD
                is_self_adjoint=True,
                is_positive_definite=True,
            )

            # observation_particles_diff = Y - G(X) - η
            observation_particles_diff = (current_observation - correct_samples)

            if observation_size_is_static_and_scalar:
                # In the univariate observation case, the Kalman gain is given by:
                # K = cov(X, Y) / (var(Y) + var_noise). That is we just divide
                # by the particle covariance plus the observation noise.
                kalman_gain = tf.nest.map_structure(
                    lambda x: x / corrected_cov,
                    covariance_xy)
                new_particles = tf.nest.map_structure(
                    lambda x, g: x + dampling * tf.linalg.matvec(  # pylint:disable=g-long-lambda
                        g, observation_particles_diff), state_prior_samples, kalman_gain)
            else:
                # added_term = [Cov(G(X)) + Γ]⁻¹ [Y - G(X) - η]
                added_term = covriance_yy.solvevec(
                    observation_particles_diff)
                # added_term
                #  = covariance_between_state_and_predicted_observations @ added_term
                #  = Cov(X, G(X)) [Cov(G(X)) + Γ]⁻¹ [Y - G(X) - η]
                #  = (X - μ(X))(G(X) - μ(G(X)))ᵀ [Cov(G(X)) + Γ]⁻¹ [Y - G(X) - η]
                added_term = tf.nest.map_structure(
                    lambda x: tf.linalg.matvec(x, added_term),
                    covariance_xy)

                # new_particles = X + damping * added_term
                new_particles = tf.nest.map_structure(
                    lambda x, a: x + dampling * a, correct_samples, added_term)

            return new_particles

        filter_result = tf.scan(ekf_one_step, observations, prior_samples)

        return tf.reduce_mean(filter_result, axis=1), tfp.stats.covariance(
                            filter_result, sample_axis=1, event_axis=-1, keepdims=False)

    # Sample covariance. Handles differing shapes.
    @staticmethod
    def _covariance(x, y=None):
        """Sample covariance, assuming samples are the leftmost axis."""
        x = tf.convert_to_tensor(x, name='x')
        # Covariance *only* uses the centered versions of x (and y).
        x = x - tf.reduce_mean(x, axis=0)

        if y is None:
            y = x
        else:
            y = tf.convert_to_tensor(y, name='y', dtype=x.dtype)
            y = y - tf.reduce_mean(y, axis=0)

        return tf.reduce_mean(tf.linalg.matmul(
            x[..., tf.newaxis],
            y[..., tf.newaxis], adjoint_b=True), axis=0)

