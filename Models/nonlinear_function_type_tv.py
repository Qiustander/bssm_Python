import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from .check_argument import *

"""
Time-varying nonlinear model, only keep the distribution object, remove the functional form

Note: need to use tf.stack/tf.concat for each dimension of observation/state in the funtional form in order to boardcast
"""

def nonlinear_fucntion(function_type,
                       obs_len,
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
    """
    Time-varying nonlinear model, insert time step t into the functional form and distribution
    Args:
      function_type: nonlinear SSM type denoted as string, currently support type:
      "nlg_linear_gaussian", "nlg_sin_exp", "nlg_growth", "nlg_ar_exp", "sde_poisson_ou", "sde_gbm"
      obs_len: the length of the time series
      state_dim: dimension of state variable
      obs_dim: dimension of observation features
      state_noise: std of state noise
      obs_noise: std of observation noise
      prior_mean (np.array): Prior mean for the initial state as a vector of length m.
      prior_cov (np.array): Prior covariance matrix for the initial state as m x m matrix.
      rho_state (np.array): Autoregressive coefficient. It is a distribution with prior value. For AR model
      mu_state: Autoregressive mean. It is a distribution with prior value. For AR model
      input_state: input state
      input_obs: input obs

      "nlg_sin_exp": univariate
          x_0 = 0
          x_t = sin(x_t-1) + N(0, state_noise)
          y_t = exp(x_t) + N(0, 0.01)
      "nlg_ar_exp": univariate
          x_0 ~ N(mu_state, sigma_x/(1-rho_state^2))
          x_t = mu_state*(1-rho_state) + rho_state*x_-1 + N(0, state_noise^2)
          y_t = exp(x_t) + N(0, 0.01)
    Returns:
        List of: [observation_fn, transition_fn,
                observation_fn_grad, transition_fn_grad,
                transition_noise_fn, observation_noise_fn, initial_state_prior]
            observation_fn: [observation_size, latent_size]
            observation_eq: [observation_size, latent_size]
            transition_fn: [latent_size, latent_size]
            observation_fn_grad: [observation_size, latent_size]
            transition_fn_grad: [latent_size, latent_size]
            transition_noise_fn: [latent_size]
            observation_noise_fn: [[observation_size]
            initial_state_prior: [latent_size]
    """

    def jacobian_fn(jaco_fucn):
        def inner_wrap(t, x):
            with tf.GradientTape() as g:
                g.watch(x)
                y = jaco_fucn(t, x)
            jaco_matrix = g.jacobian(y, x)
            return jaco_matrix
        return inner_wrap

    try:
        if function_type == "nlg_sin_exp":

            input_obs = check_input_obs(0., obs_dim, obs_len) if not isinstance(input_obs, np.ndarray) else \
                    check_input_obs(input_obs, obs_dim, obs_len)
            input_state = check_input_state(0., state_dim, obs_len) if not isinstance(input_state, np.ndarray) else \
                    check_input_state(input_state, state_dim, obs_len)

            observation_fn = lambda t, x: tf.cast(tf.stack([tf.exp(x[..., 0])], axis=-1), dtype=dtype)
            observation_dist = lambda t, x: tfd.MultivariateNormalLinearOperator(
                        loc=observation_fn(t, x) + tf.convert_to_tensor(input_obs, dtype=dtype),
                    scale=tf.linalg.LinearOperatorFullMatrix(tf.cast(
                        check_obs_mtx_noise(obs_noise, obs_dim, obs_len), dtype=dtype)))

            transition_fn = lambda t, x: tf.cast(tf.stack([tf.sin(x[..., 0])], axis=-1), dtype=dtype)
            transition_dist = lambda t, x: tfd.MultivariateNormalLinearOperator(
                        loc=transition_fn(t, x) + tf.convert_to_tensor(input_state, dtype=dtype),
                    scale=tf.linalg.LinearOperatorFullMatrix(tf.cast(
                        check_state_noise(state_noise, state_dim, obs_len), dtype=dtype)))

            observation_fn_grad = jacobian_fn(observation_fn)
            transition_fn_grad = jacobian_fn(transition_fn)

            prior_mean = tf.convert_to_tensor(check_prior_mean(prior_mean, state_dim), dtype=dtype)
            prior_cov = tf.convert_to_tensor(check_prior_cov(prior_cov, state_dim), dtype=dtype)
            # DO NOT use LinearOperatorDiag, it would add one dimension
            initial_state_prior = tfd.MultivariateNormalLinearOperator(
                                     loc=prior_mean,
                                        scale=tf.linalg.LinearOperatorFullMatrix(tf.sqrt(prior_cov)) if tf.size(prior_cov) == 1 else
                                        tf.linalg.LinearOperatorLowerTriangular(tf.linalg.cholesky(prior_cov)))

        elif function_type == "nlg_ar_exp":
            if not rho_state or not mu_state:
                raise AssertionError("Not defined mu and rho for AR model!")
            rho_state = check_rho(rho_state)
            mu_state = check_mu(mu_state)

            input_obs = check_input_obs(0., obs_dim, obs_len) if not isinstance(input_obs, np.ndarray) else \
                    check_input_obs(input_obs, obs_dim, obs_len)
            input_state = check_input_state(0., state_dim, obs_len) if not isinstance(input_state, np.ndarray) else \
                    check_input_state(input_state, state_dim, obs_len)

            observation_fn = lambda t, x: tf.cast(tf.stack([tf.exp(x[..., 0])], axis=-1), dtype=dtype)
            observation_dist = lambda t, x: tfd.MultivariateNormalLinearOperator(
                        loc=observation_fn(t, x) + tf.convert_to_tensor(input_obs, dtype=dtype),
                    scale=tf.linalg.LinearOperatorFullMatrix(tf.cast(
                        check_obs_mtx_noise(obs_noise, obs_dim, obs_len), dtype=dtype)))

            transition_fn = lambda t, x: tf.cast(tf.stack([mu_state*(1 -rho_state) + rho_state*x[..., 0]], axis=-1), dtype=dtype)
            transition_dist = lambda t, x: tfd.MultivariateNormalLinearOperator(
                        loc=transition_fn(t, x) + tf.convert_to_tensor(input_state, dtype=dtype),
                    scale=tf.linalg.LinearOperatorFullMatrix(tf.cast(
                        check_state_noise(state_noise, state_dim, obs_len), dtype=dtype)))

            transition_fn_grad = jacobian_fn(transition_fn)
            observation_fn_grad = jacobian_fn(observation_fn)

            prior_mean = tf.convert_to_tensor(check_prior_mean(mu_state, state_dim), dtype=dtype)
            prior_cov = tf.convert_to_tensor(check_prior_cov(state_noise**2/(1 - rho_state**2), state_dim), dtype=dtype)

            initial_state_prior = tfd.MultivariateNormalLinearOperator(
                                     loc=prior_mean,
                                        scale=tf.linalg.LinearOperatorFullMatrix(tf.sqrt(prior_cov)) if tf.size(prior_cov) == 1 else
                                        tf.linalg.LinearOperatorLowerTriangular(tf.linalg.cholesky(prior_cov)))

        elif function_type == "nlg_mv_model":

            input_obs = check_input_obs(0., obs_dim, obs_len) if not isinstance(input_obs, np.ndarray) else \
                    check_input_obs(input_obs, obs_dim, obs_len)
            input_state = check_input_state(0., state_dim, obs_len) if not isinstance(input_state, np.ndarray) else \
                    check_input_state(input_state, state_dim, obs_len)

            # state dim = 4, obs dim = 3
            observation_fn = lambda t, x: tf.cast(tf.stack([x[..., 0]**2, x[..., 1]**3,
                                        0.5*x[..., 2] + 2*x[..., 3] + x[..., 0] + x[..., 1]], axis=-1), dtype=dtype)
            observation_dist = lambda t, x: tfd.MultivariateNormalLinearOperator(
                        loc=observation_fn(t, x) + tf.convert_to_tensor(input_obs, dtype=dtype),
                    scale=tf.linalg.LinearOperatorFullMatrix(tf.cast(
                        check_obs_mtx_noise(obs_noise, obs_dim, obs_len), dtype=dtype)))

            transition_fn = lambda t, x: tf.cast(tf.stack([0.8*x[..., 0] + kwargs['dt']*x[..., 1], 0.7*x[..., 1] + kwargs['dt']*x[..., 2],
                                        0.6*x[..., 2] + kwargs['dt']*x[..., 3], 0.6*x[..., 3] + kwargs['dt']*x[..., 0]], axis=-1), dtype=dtype)
            transition_dist = lambda t, x: tfd.MultivariateNormalLinearOperator(
                        loc=transition_fn(t, x) + tf.convert_to_tensor(input_state, dtype=dtype),
                    scale=tf.linalg.LinearOperatorFullMatrix(tf.cast(
                        check_state_noise(state_noise, state_dim, obs_len), dtype=dtype)))

            transition_fn_grad = jacobian_fn(transition_fn)
            observation_fn_grad = jacobian_fn(observation_fn)

            prior_mean = tf.convert_to_tensor(check_prior_mean(prior_mean, state_dim), dtype=dtype)
            prior_cov = tf.convert_to_tensor(check_prior_cov(prior_cov, state_dim), dtype=dtype)
            # DO NOT use LinearOperatorDiag, it would add one dimension
            initial_state_prior = tfd.MultivariateNormalLinearOperator(
                                     loc=prior_mean,
                                        scale=tf.linalg.LinearOperatorFullMatrix(tf.sqrt(prior_cov)) if tf.size(prior_cov) == 1 else
                                        tf.linalg.LinearOperatorLowerTriangular(tf.linalg.cholesky(prior_cov)))

        elif function_type == "constant_dynamic_univariate_test":

            input_obs = check_input_obs(0., obs_dim, obs_len) if not isinstance(input_obs, np.ndarray) else \
                    check_input_obs(input_obs, obs_dim, obs_len)
            input_state = check_input_state(0., state_dim, obs_len) if not isinstance(input_state, np.ndarray) else \
                    check_input_state(input_state, state_dim, obs_len)

            observation_fn = lambda t, x: tf.cast(tf.stack([x[..., 0]], axis=-1), dtype=dtype)
            observation_dist = lambda t, x: tfd.MultivariateNormalLinearOperator(
                        loc=observation_fn(t, x) + tf.convert_to_tensor(input_obs, dtype=dtype),
                    scale=tf.linalg.LinearOperatorFullMatrix(tf.cast(
                        check_obs_mtx_noise(obs_noise, obs_dim, obs_len), dtype=dtype)))

            transition_fn = lambda t, x: tf.cast(tf.stack([x[..., 0]], axis=-1), dtype=dtype)
            transition_dist = lambda t, x: tfd.MultivariateNormalLinearOperator(
                        loc=transition_fn(t, x) + tf.convert_to_tensor(input_state, dtype=dtype),
                    scale=tf.linalg.LinearOperatorFullMatrix(tf.cast(
                        check_state_noise(state_noise, state_dim, obs_len), dtype=dtype)))

            transition_fn_grad = jacobian_fn(transition_fn)
            observation_fn_grad = jacobian_fn(observation_fn)

            prior_mean = tf.convert_to_tensor(check_prior_mean(prior_mean, state_dim), dtype=dtype)
            prior_cov = tf.convert_to_tensor(check_prior_cov(prior_cov, state_dim), dtype=dtype)
            # DO NOT use LinearOperatorDiag, it would add one dimension
            initial_state_prior = tfd.MultivariateNormalLinearOperator(
                                     loc=prior_mean,
                                        scale=tf.linalg.LinearOperatorFullMatrix(tf.sqrt(prior_cov)) if tf.size(prior_cov) == 1 else
                                        tf.linalg.LinearOperatorLowerTriangular(tf.linalg.cholesky(prior_cov)))

        elif function_type == "constant_dynamic_multivariate_test":

            input_obs = check_input_obs(0., obs_dim, obs_len) if not isinstance(input_obs, np.ndarray) else \
                    check_input_obs(input_obs, obs_dim, obs_len)
            input_state = check_input_state(0., state_dim, obs_len) if not isinstance(input_state, np.ndarray) else \
                    check_input_state(input_state, state_dim, obs_len)

            observation_fn = lambda t, x: tf.cast(tf.stack([x[..., 0], x[..., 1],
                                        x[..., 2] + 0.1* x[..., 3]], axis=-1), dtype=dtype)
            observation_dist = lambda t, x: tfd.MultivariateNormalLinearOperator(
                        loc=observation_fn(t, x) + tf.convert_to_tensor(input_obs, dtype=dtype),
                    scale=tf.linalg.LinearOperatorFullMatrix(tf.cast(
                        check_obs_mtx_noise(obs_noise, obs_dim, obs_len), dtype=dtype)))

            transition_fn = lambda t, x: tf.cast(tf.stack([x[..., 0], x[..., 1],
                                       x[..., 2], x[..., 3]], axis=-1), dtype=dtype)
            transition_dist = lambda t, x: tfd.MultivariateNormalLinearOperator(
                        loc=transition_fn(t, x) + tf.convert_to_tensor(input_state, dtype=dtype),
                    scale=tf.linalg.LinearOperatorFullMatrix(tf.cast(
                        check_state_noise(state_noise, state_dim, obs_len), dtype=dtype)))

            transition_fn_grad = jacobian_fn(transition_fn)
            observation_fn_grad = jacobian_fn(observation_fn)

            prior_mean = tf.convert_to_tensor(check_prior_mean(prior_mean, state_dim), dtype=dtype)
            prior_cov = tf.convert_to_tensor(check_prior_cov(prior_cov, state_dim), dtype=dtype)
            # DO NOT use LinearOperatorDiag, it would add one dimension
            initial_state_prior = tfd.MultivariateNormalLinearOperator(
                                     loc=prior_mean,
                                        scale=tf.linalg.LinearOperatorFullMatrix(tf.sqrt(prior_cov)) if tf.size(prior_cov) == 1 else
                                        tf.linalg.LinearOperatorLowerTriangular(tf.linalg.cholesky(prior_cov)))

        return {"observation_dist": observation_dist, "transition_dist": transition_dist,
                "observation_fn_grad": observation_fn_grad, "transition_fn_grad": transition_fn_grad,
                "observation_fn": observation_fn, "transition_fn": transition_fn,
                "initial_state_prior": initial_state_prior, "num_timesteps": obs_len, "state_dim": state_dim}

    except:
        raise AttributeError("No nonlinear function is found! Please define a specific one.")