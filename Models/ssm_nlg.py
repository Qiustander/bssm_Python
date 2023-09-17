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

    def __init__(self,
                 num_timesteps,
                 state_dim,
                 observation_dist,
                 transition_dist,
                 transition_fn,
                 observation_fn,
                 observation_fn_grad,
                 transition_fn_grad,
                 initial_state_prior,
                 initial_step=0,
                 mask=None,
                 experimental_parallelize=False,
                 validate_args=False,
                 allow_nan_stats=True,
                 dtype=tf.float32,
                 seed=None,
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
            self._observation_dist = observation_dist
            self._transition_dist = transition_dist
            self._transition_fn = transition_fn
            self._observation_fn = observation_fn
            self.seed = seed

            dtype_list = [initial_state_prior,
                          observation_dist,
                          transition_dist,
                          observation_fn_grad,
                          transition_fn_grad,
                          transition_fn,
                          observation_fn]

            # Infer dtype from time invariant objects. This list will be non-empty
            # since it will always include `initial_state_prior`.
            dtype = dtype_util.common_dtype(
                list(filter(lambda x: not callable(x), dtype_list)),
                dtype_hint=dtype)
            # self.dtype = dtype

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
    def transition_dist(self):
        return self._transition_dist

    @property
    def observation_fn(self):
        return self._observation_fn

    @property
    def transition_fn(self):
        return self._transition_fn

    @property
    def observation_dist(self):
        return self._observation_dist

    @property
    def initial_state_prior(self):
        return self._initial_state_prior


    # Function type

    def auxiliary_fn(self):
        """Auxiliary function for the Auxiliary Particle Filter, Default None
        """
        pass

    def log_target_dist(self):
        """ Target (Log) Distribution for the Markov Chain Monte Carlo, Default None
        The target distribution should be the log_prob attribute of a distribution
        Args:
            observations: the input observations y_{1:T}
            num_particles: for particle MCMC
        Returns:

        """
        pass

    def log_theta_prior(self):
        """Log Prior likelihood of the parameters theta
        Args:
            theta: parameters
        Returns: log_prob of the prior distritbuion

        """
        pass

    def proposal_cond_list(self):
        """List of conditional distribution for the Gibbs sampling, Default None
        a list of tuples `(state_part_idx, kernel_make_fn)`.
                            `state_part_idx` denotes the index (relative to
                            positional args in `target_log_prob_fn`) of the
                            state the kernel updates.  `kernel_make_fn` takes
                            arguments `target_log_prob_fn` and `state`, returning
                            a `tfp.mcmc.TransitionKernel`.
        """

        pass

    def psi_two_filter_fn(self):
        """Artificial distribution for the generalized two-filter Particle smoother, Default None
        """
        pass

    def proposal_dist(self):
        """Proposal distribution for and SMC, Default None
        """
        raise NotImplementedError

    def update_model(self):
        """Update the current function related to the parameters theta
        """
        raise NotImplementedError

    def initial_theta(self):
        """Initialize the parameters theta for parameterization
        """
        pass

    @tf.function
    def simulate(self, len_time_step=None, seed=None):
        """
        Simulate true state and observations for filtering and smoothing, and parameter estimation.
        Args:
            seed: generated seed
        Returns:
            observations
        """

        if len_time_step is None:
            len_time_step = self._num_timesteps

        def _generate_signal(transition_fn, observation_fn):
            def _inner_wrap(gen_data, current_step):
                last_state, _ = gen_data

                current_state = transition_fn(current_step, last_state).sample(seed=seed)
                current_observation = observation_fn(current_step, current_state).sample(seed=seed)

                return current_state, current_observation

            return _inner_wrap

        gen_data = _generate_signal(self._transition_dist, self._observation_dist)

        initial_state = self._initial_state_prior.sample()
        init_obs = self._observation_dist(0, initial_state).sample()
        overall_step = tf.range(1, len_time_step)

        true_state, observations = tf.scan(gen_data,
                                           elems=overall_step,
                                           initializer=(initial_state,
                                                        init_obs),
                                           )
        true_state = tf.concat([initial_state[tf.newaxis], true_state], axis=0)
        observations = tf.concat([init_obs[tf.newaxis], observations], axis=0)

        return true_state, observations

    # @classmethod
    # def copy_instance(cls, original_instance):
    #     new_instance = cls.__new__(cls)  # Create a 'blank' instance
    #     new_instance.__dict__ = original_instance.__dict__.copy()  # Copy attributes
    #
    #     # for attr_name, attr_value in original_instance.__dict__.items():
    #     #     setattr(self, attr_name, attr_value)
    #     for attr_name, attr_value in original_instance.__class__.__dict__.items():
    #         if isinstance(attr_value, property):
    #             setattr(NewClassInstance, attr_name, property(attr_value.fget, attr_value.fset, attr_value.fdel))
    #

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
                     dtype=tf.float32,
                     **kwargs):
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
            dtype=dtype,
            **kwargs))
