import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from .check_argument import *

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
        List of: [observation_fn_fn, transition_fn_fn,
                observation_fn_grad, transition_fn_grad,
                transition_noise_fn, observation_noise_fn, initial_state_prior]
            observation_fn_fn: [observation_size, latent_size]
            transition_fn_fn: [latent_size, latent_size]
            observation_fn_grad: [observation_size, latent_size]
            transition_fn_grad: [latent_size, latent_size]
            transition_noise_fn: [latent_size]
            observation_noise_fn: [[observation_size]
            initial_state_prior: [latent_size]
    """
    # TODO : time varying

    try:
        if function_type == "nlg_sin_exp":

            observation_fn = lambda x: tfd.MultivariateNormalLinearOperator(
                        loc=tf.exp(x[..., 0])*tf.ones([obs_dim], dtype=dtype),
                    scale=tf.linalg.LinearOperatorFullMatrix(tf.cast(
                        check_obs_mtx_noise(obs_noise, obs_dim, obs_len), dtype=dtype)))
            transition_fn = lambda x: tfd.MultivariateNormalLinearOperator(
                        loc=tf.sin(x[..., 0])*tf.ones([state_dim], dtype=dtype),
                    scale=tf.linalg.LinearOperatorFullMatrix(tf.cast(
                        check_state_noise(state_noise, state_dim, obs_len), dtype=dtype)))
            observation_fn_grad = lambda x: tf.reshape(tf.exp(x[..., 0]), [state_dim, state_dim])
            transition_fn_grad = lambda x: tf.reshape(tf.cos(x[..., 0]), [obs_dim, state_dim])

            # The transition_noise_fn and observation_noise_fn are contained in the obs_fn
            # and state_fn for using extended Kalman filter. So only for definition for future usage
            if not input_obs:
                input_obs = check_input_obs(0., obs_dim, obs_len)
            if not input_state:
                input_state = check_input_state(0., state_dim, obs_len)
            transition_noise_fn = lambda x: tfd.MultivariateNormalLinearOperator(
                        loc=tf.convert_to_tensor(input_state, dtype=dtype),
                    scale=tf.linalg.LinearOperatorFullMatrix(tf.cast(
                        check_state_noise(state_noise, state_dim, obs_len), dtype=dtype)))
            observation_noise_fn = lambda x: tfd.MultivariateNormalLinearOperator(
                             loc=tf.convert_to_tensor(input_obs, dtype=dtype),
                            scale= tf.linalg.LinearOperatorFullMatrix(
                                check_obs_mtx_noise(obs_noise, obs_dim, obs_len), dtype=dtype
                            ))

            prior_mean = tf.convert_to_tensor(check_prior_mean(prior_mean, state_dim), dtype=dtype)
            prior_cov = tf.convert_to_tensor(check_prior_cov(prior_cov, state_dim), dtype=dtype)
            # TODO: add to unittest! must take care of batch shape problem!
            # DO NOT use LinearOperatorDiag, it would add one dimension
            initial_state_prior = tfd.MultivariateNormalLinearOperator(
                                     loc=prior_mean,
                                        scale=tf.linalg.LinearOperatorFullMatrix(tf.sqrt(prior_cov)) if tf.size(prior_cov) == 1 else
                                        tf.linalg.LinearOperatorLowerTriangular(tf.linalg.cholesky(prior_cov)))

            return {"observation_fn":observation_fn, "transition_fn":transition_fn,
               "observation_fn_grad":observation_fn_grad, "transition_fn_grad":transition_fn_grad,
               "transition_noise_fn":transition_noise_fn, "observation_noise_fn":observation_noise_fn,
                    "initial_state_prior":initial_state_prior, "num_timesteps":obs_len}

        elif function_type == "nlg_ar_exp":
            if not rho_state or not mu_state:
                raise AssertionError("Not defined mu and rho for AR model!")
            rho_state = check_rho(rho_state)
            mu_state = check_mu(mu_state)

            observation_fn = lambda x: tfd.MultivariateNormalLinearOperator(
                        loc=tf.exp(x[..., 0])*tf.ones([obs_dim], dtype=dtype),
                    scale=tf.linalg.LinearOperatorFullMatrix(tf.cast(
                        check_obs_mtx_noise(obs_noise, obs_dim, obs_len), dtype=dtype)))
            transition_fn = lambda x: tfd.MultivariateNormalLinearOperator(
                        loc=(mu_state*(1 -rho_state) + rho_state*x[..., 0])*tf.ones([state_dim], dtype=dtype),
                    scale=tf.linalg.LinearOperatorFullMatrix(tf.cast(
                        check_state_noise(state_noise, state_dim, obs_len), dtype=dtype)))
            observation_fn_grad = lambda x: tf.reshape(tf.exp(x[..., 0]), [state_dim, state_dim])
            transition_fn_grad = lambda x: tf.reshape(tf.cast(rho_state, dtype=dtype), [obs_dim, state_dim])

            # The transition_noise_fn and observation_noise_fn are contained in the obs_fn
            # and state_fn for using extended Kalman filter. So only for definition for future usage
            if not input_obs:
                input_obs = check_input_obs(0, obs_dim, obs_len)
            if not input_state:
                input_state = check_input_state(0, state_dim, obs_len)
            transition_noise_fn = lambda x: tfd.MultivariateNormalLinearOperator(
                        loc=tf.convert_to_tensor(input_state, dtype=dtype),
                    scale=tf.linalg.LinearOperatorFullMatrix(
                        check_state_noise(state_noise, state_dim, obs_len), dtype=dtype))
            observation_noise_fn = lambda x: tfd.MultivariateNormalLinearOperator(
                             loc=tf.convert_to_tensor(input_obs),
                            scale= tf.linalg.LinearOperatorFullMatrix(
                                check_obs_mtx_noise(obs_noise, obs_dim, obs_len), dtype=dtype
                            ))

            prior_mean = tf.convert_to_tensor(check_prior_mean(mu_state, state_dim), dtype=dtype)
            prior_cov = tf.convert_to_tensor(check_prior_cov(state_noise**2/(1 - rho_state**2), state_dim), dtype=dtype)

            initial_state_prior = tfd.MultivariateNormalLinearOperator(
                                     loc=prior_mean,
                                        scale=tf.linalg.LinearOperatorFullMatrix(tf.sqrt(prior_cov)) if tf.size(prior_cov) == 1 else
                                        tf.linalg.LinearOperatorLowerTriangular(tf.linalg.cholesky(prior_cov)))

            return {"observation_fn":observation_fn, "transition_fn":transition_fn,
               "observation_fn_grad":observation_fn_grad, "transition_fn_grad":transition_fn_grad,
               "transition_noise_fn":transition_noise_fn, "observation_noise_fn":observation_noise_fn,
                    "initial_state_prior":initial_state_prior, "num_timesteps":obs_len}

    except:
        raise AttributeError("No nonlinear function is found! Please define a specific one.")
