import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.distributions import mvn_tril

tfd = tfp.distributions


def ensemble_kalman_filter(ssm_model, observations, num_particles, dampling=1):
    """Applies an Ensemble Kalman Filter to observed data.

    [1] Geir Evensen. Sequential data assimilation with a nonlinear
    quasi-geostrophic model using Monte Carlo methods to forecast error
    statistics. Journal of Geophysical Research, 1994.

    [2] Matthias Katzfuss, Jonathan R. Stroud & Christopher K. Wikle
    Understanding the Ensemble Kalman Filter.
    The Americal Statistician, 2016.

    [3] Jeffrey L. Anderson and Stephen L. Anderson. A Monte Carlo Implementation
    of the Nonlinear Filtering Problem to Produce Ensemble Assimilations and
    Forecasts. Monthly Weather Review, 1999.

    Args:
    ssm_model: state space model object
    observations: a (structure of) `Tensor`s, each of shape
        `concat([[num_timesteps, b1, ..., bN], [event_size]])` with scalar
        `event_size` and optional batch dimensions `b1, ..., bN`.
    num_particles: number of ensembles at each step. Could be time-varying
    dampling: Floating-point `Tensor` representing how much to damp the
            update by. Used to mitigate filter divergence. Default value: 1.

    Returns:
      filtered_mean: a (structure of) `Tensor`(s) of shape
        `concat([[num_timesteps, b1, ..., bN], [state_size]])`. The mean of the
        filtered state estimate.
      filtered_cov: a (structure of) `Tensor`(s) of shape
        `concat([[num_timesteps, b1, ..., bN], [state_size, state_size]])`.
         The covariance of the filtered state estimate.
      predicted_mean: a (structure of) `Tensor`(s) of shape
        `concat([[num_timesteps, b1, ..., bN], [state_size]])`. The prior
        predicted means of the state.
      predicted_cov: a (structure of) `Tensor`(s) of shape
        `concat([[num_timesteps, b1, ..., bN], [state_size, state_size]])`
        The prior predicted covariances of the state estimate.
      observation_mean: a (structure of) `Tensor`(s) of shape
        `concat([[num_timesteps, b1, ..., bN], [event_size]])`. The prior
        predicted mean of observations.
      observation_cov: a (structure of) `Tensor`(s) of shape
        `concat([[num_timesteps, b1, ..., bN], [event_size, event_size]])`. The
        prior predicted covariance of observations.
        """

    prior_samples = ssm_model.initial_state_prior.sample(num_particles)

    (filtered_particles,
     predicted_particles,
     log_marginal_likelihood) = forward_filter_pass(
        transition_fn=ssm_model.transition_dist,
        observation_fn=ssm_model.observation_dist,
        observations=observations,
        initial_particles=prior_samples,
        scaling_parameters=(dampling))

    filtered_means = tf.reduce_mean(filtered_particles, axis=1)
    filtered_covs = tfp.stats.covariance(
        filtered_particles, sample_axis=1, event_axis=-1, keepdims=False)

    # one-step prediction
    state_prior_samples = tf.vectorized_map(lambda x:
                                            ssm_model.transition_dist(prefer_static.size0(observations) - 1,
                                                                      x).sample(),
                                            filtered_particles[-1, ...])[tf.newaxis]
    predicted_means = tf.reduce_mean(
        tf.concat([predicted_particles, state_prior_samples], axis=0), axis=1)
    predicted_covs = tfp.stats.covariance(
        state_prior_samples, sample_axis=1, event_axis=-1, keepdims=False)

    return (filtered_means, filtered_covs,
            predicted_means, predicted_covs, log_marginal_likelihood)


def forward_filter_pass(transition_fn,
                        observation_fn,
                        scaling_parameters,
                        observations,
                        initial_particles):
    """Run the forward pass in ensembles Kalman filter.

    Args:
      observations:
      scaling_parameters:
      transition_fn: a Python `callable` that accepts (batched) vectors of length
        `state_size`, and returns a `tfd.Distribution` instance, typically a
        `MultivariateNormal`, representing the state transition and covariance.
      observation_fn: a Python `callable` that accepts a (batched) vector of
        length `state_size` and returns a `tfd.Distribution` instance, typically
        a `MultivariateNormal` representing the observation model and covariance.

    Returns:
        filtered_particles
    """
    update_step_fn = build_forward_filter_step(
        transition_fn,
        observation_fn,
        scaling_parameters)

    observations_shape = prefer_static.shape(
        tf.nest.flatten(observations)[0])
    dummy_zeros = tf.zeros(observations_shape[1:-1])

    (filtered_particles,
     predicted_particles,
     log_marginal_likelihood, _) = tf.scan(update_step_fn,
                                           elems=observations,
                                           initializer=(initial_particles,
                                                        initial_particles,
                                                        dummy_zeros,
                                                        dummy_zeros))

    return (filtered_particles,
            predicted_particles,
            log_marginal_likelihood)


def build_forward_filter_step(transition_fn,
                              observation_fn,
                              scaling_parameters):
    """Build a callable that perform one step for backward smoothing.

    Args:

    Returns:
    backward_pass_step: a callable that updates a BackwardPassState
      from timestep `t` to `t-1`.
    """

    def forward_pass_step(filtered_ensembles,
                          observations):
        """Run a single step of backward smoothing."""
        dampling = scaling_parameters

        (filtered_particles,
         predicted_particles,
         log_marginal_likelihood,
         time_step) = _ensemble_kalman_filter_one_step(filtered_ensembles, observations,
                                                       transition_fn=transition_fn,
                                                       observation_fn=observation_fn,
                                                       dampling=dampling)

        return (filtered_particles,
                predicted_particles,
                log_marginal_likelihood,
                time_step)

    return forward_pass_step


def _ensemble_kalman_filter_one_step(
        filtered_ensembles, observation, transition_fn, observation_fn, dampling):
    """A single step of the EnKF.

    Args:
    state: A `Tensor` of shape
      `concat([[num_timesteps, b1, ..., bN], [state_size]])` with scalar
      `event_size` and optional batch dimensions `b1, ..., bN`.
    observation: A `Tensor` of shape
      `concat([[num_timesteps, b1, ..., bN], [event_size]])` with scalar
      `event_size` and optional batch dimensions `b1, ..., bN`.
    transition_fn: a Python `callable` that accepts (batched) vectors of length
      `state_size`, and returns a `tfd.Distribution` instance, typically a
      `MultivariateNormal`, representing the state transition and covariance.
    observation_fn: a Python `callable` that accepts a (batched) vector of
      length `state_size` and returns a `tfd.Distribution` instance, typically
      a `MultivariateNormal` representing the observation model and covariance.
    Returns:
    updated_state: filtered ensembles
    """
    time_step = filtered_ensembles[-1]

    # If observations are scalar, we can avoid some matrix ops.
    observation_size_is_static_and_scalar = (observation.shape[-1] == 1)

    ############### Estimation
    state_prior_samples = tf.vectorized_map(lambda x:
                                            transition_fn(time_step, x).sample(), filtered_ensembles[0])

    ########### Correction
    correct_samples = tf.vectorized_map(lambda x:
                                        observation_fn(time_step, x).sample(), state_prior_samples)

    # corrected_mean = tf.reduce_mean(correct_samples)
    corrected_cov = _covariance(correct_samples)

    # covariance_between_state_and_predicted_observations
    # Cov(X, G(X))  = (X - μ(X))(G(X) - μ(G(X)))ᵀ
    covariance_xy = tf.nest.map_structure(
        lambda x: _covariance(x, correct_samples),
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
    observation_particles_diff = (observation - correct_samples)

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
            lambda x, a: x + dampling * a, state_prior_samples, added_term)

    predicted_observations = observation_fn(time_step, state_prior_samples).mean()
    observation_dist = mvn_tril.MultivariateNormalTriL(
        loc=tf.reduce_mean(predicted_observations, axis=0),  # ensemble mean
        # Cholesky(Cov(G(X)) + Γ), where Cov(..) is the ensemble covariance.
        scale_tril=tf.linalg.cholesky(
            _covariance(predicted_observations) +
            _linop_covariance(observation_fn(time_step, state_prior_samples)).to_dense()))

    log_marginal_likelihood = observation_dist.log_prob(observation)

    return (new_particles,
            state_prior_samples,
            log_marginal_likelihood,
            time_step + 1)


def _linop_covariance(dist):
    """LinearOperator backing Cov(dist), without unnecessary broadcasting."""
    # Simply calling dist.covariance() would broadcast up to the full batch shape.
    # Instead, we want the shape to be that of the linear operator only.
    # This (i) saves memory and (ii) allows operations done with this operator
    # to be more efficient.
    if hasattr(dist, 'cov_operator'):
        cov = dist.cov_operator
    else:
        cov = dist.scale.matmul(dist.scale.H)
    cov._is_positive_definite = True  # pylint: disable=protected-access
    cov._is_self_adjoint = True  # pylint: disable=protected-access
    return cov


def _covariance(x, y=None):
    """Sample covariance, assuming samples are the leftmost axis."""
    x = tf.convert_to_tensor(x, name='x')
    # Covariance *only* uses the centered versions of x (and y).
    x = x - tf.reduce_mean(x, axis=0)

    if y is None:
        y = x
    else:
        y = tf.convert_to_tensor(y, name='y', dtype=x.dtype)

    return tf.reduce_mean(tf.linalg.matmul(
        x[..., tf.newaxis],
        y[..., tf.newaxis], adjoint_b=True), axis=0)
