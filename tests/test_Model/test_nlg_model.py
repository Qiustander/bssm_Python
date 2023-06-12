import pytest
import tensorflow as tf
import numpy as np
from Models.ssm_nlg import NonlinearSSM


class TestNonlinearModel:

    def test_multivariate_model_shape(self):
        # State dim 4, observation dim 3
        num_timesteps = 20
        state_dim = 4
        observed_dim = 3
        prior_mean = np.ones(shape=state_dim)
        prior_cov = np.ones(shape=[state_dim, state_dim])
        input_state = np.random.randn(state_dim,)
        input_obs = np.random.randn(observed_dim,)
        state_mtx_noise = np.random.randn(state_dim, state_dim)
        obs_mtx_noise = np.random.randn(observed_dim, observed_dim)
        test_state = tf.convert_to_tensor(np.random.randn(state_dim, ), dtype=tf.float32)

        model_obj = NonlinearSSM.create_model(num_timesteps=num_timesteps,
                                              observation_size=observed_dim,
                                              latent_size=state_dim,
                                              initial_state_mean=prior_mean,
                                              initial_state_cov=prior_cov,
                                              state_noise_std=state_mtx_noise,
                                              input_obs=input_obs,
                                              input_state=input_state,
                                              obs_noise_std=obs_mtx_noise,
                                              dt=0.3,
                                              nonlinear_type="nlg_mv_model")
        tf.debugging.assert_shapes([(model_obj.observation_fn(test_state), (observed_dim, )),
                                    (model_obj.transition_fn(test_state), (state_dim,)),
                                    (model_obj.transition_fn_grad(test_state), (state_dim, state_dim)),
                                    (model_obj.observation_fn_grad(test_state), (observed_dim, state_dim)),
                                    (model_obj.transition_noise_fn.sample(), (state_dim,)),
                                    (model_obj.observation_noise_fn.sample(), (observed_dim,)),
                                    (model_obj.observation_plusnoise_fn(test_state).sample(), (observed_dim,)),
                                    (model_obj.transition_plusnoise_fn(test_state).sample(), (state_dim,)),
                                    (model_obj.initial_state_prior.sample(), (state_dim,)),
                                    ])

        dtype_tensor = model_obj.transition_noise_fn.mean()
        tf.debugging.assert_equal(
            model_obj.transition_noise_fn.mean(),
            tf.convert_to_tensor(input_state, dtype=dtype_tensor.dtype),
        )
        tf.debugging.assert_equal(
            model_obj.observation_noise_fn.mean(),
            tf.convert_to_tensor(input_obs, dtype=dtype_tensor.dtype),
        )

        tf.debugging.assert_near(
            model_obj.transition_noise_fn.covariance(), tf.convert_to_tensor(
                state_mtx_noise@ np.transpose(state_mtx_noise),
                dtype=dtype_tensor.dtype),
            atol=1e-4)
        tf.debugging.assert_near(
            model_obj.observation_noise_fn.covariance(), tf.convert_to_tensor(
                obs_mtx_noise @ np.transpose(obs_mtx_noise), dtype=dtype_tensor.dtype),
            atol=1e-4)

        # Assert plusnoise_fn
        tf.debugging.assert_equal(
            model_obj.observation_plusnoise_fn(test_state).mean(),
            tf.convert_to_tensor(model_obj.observation_fn(test_state), dtype=dtype_tensor.dtype),
        )
        tf.debugging.assert_equal(
            model_obj.transition_plusnoise_fn(test_state).mean(),
            tf.convert_to_tensor(model_obj.transition_fn(test_state), dtype=dtype_tensor.dtype),
        )

        tf.debugging.assert_near(
            model_obj.transition_plusnoise_fn(test_state).covariance(), tf.convert_to_tensor(
                state_mtx_noise @ np.transpose(state_mtx_noise),
                dtype=dtype_tensor.dtype),
            atol=1e-4)
        tf.debugging.assert_near(
            model_obj.observation_plusnoise_fn(test_state).covariance(), tf.convert_to_tensor(
                obs_mtx_noise @ np.transpose(obs_mtx_noise), dtype=dtype_tensor.dtype),
            atol=1e-4)

    def test_univariate_model_shape(self):
        # State dim 1, observation dim 1
        num_timesteps = 20
        state_dim = 1
        observed_dim = 1
        prior_mean = np.ones(shape=state_dim)
        prior_cov = np.ones(shape=[state_dim, state_dim])
        input_state = np.random.randn(state_dim,)
        input_obs = np.random.randn(observed_dim,)
        state_mtx_noise = np.random.randn(state_dim, state_dim)
        obs_mtx_noise = np.random.randn(observed_dim, observed_dim)
        test_state = tf.convert_to_tensor(np.random.randn(state_dim, ), dtype=tf.float32)

        model_obj = NonlinearSSM.create_model(num_timesteps=num_timesteps,
                                              observation_size=observed_dim,
                                              latent_size=state_dim,
                                              initial_state_mean=prior_mean,
                                              initial_state_cov=prior_cov,
                                              state_noise_std=state_mtx_noise,
                                              input_obs=input_obs,
                                              input_state=input_state,
                                              obs_noise_std=obs_mtx_noise,
                                              nonlinear_type="nlg_sin_exp")
        tf.debugging.assert_shapes([(model_obj.observation_fn(test_state), (observed_dim, )),
                                    (model_obj.transition_fn(test_state), (state_dim,)),
                                    (model_obj.transition_fn_grad(test_state), (state_dim, state_dim)),
                                    (model_obj.observation_fn_grad(test_state), (observed_dim, state_dim)),
                                    (model_obj.transition_noise_fn.sample(), (state_dim,)),
                                    (model_obj.observation_noise_fn.sample(), (observed_dim,)),
                                    (model_obj.observation_plusnoise_fn(test_state).sample(), (observed_dim,)),
                                    (model_obj.transition_plusnoise_fn(test_state).sample(), (state_dim,)),
                                    (model_obj.initial_state_prior.sample(), (state_dim,)),
                                    ])

        dtype_tensor = model_obj.transition_noise_fn.mean()
        tf.debugging.assert_equal(
            model_obj.transition_noise_fn.mean(),
            tf.convert_to_tensor(input_state, dtype=dtype_tensor.dtype),
        )
        tf.debugging.assert_equal(
            model_obj.observation_noise_fn.mean(),
            tf.convert_to_tensor(input_obs, dtype=dtype_tensor.dtype),
        )

        tf.debugging.assert_near(
            model_obj.transition_noise_fn.covariance(), tf.convert_to_tensor(
                state_mtx_noise@ np.transpose(state_mtx_noise),
                dtype=dtype_tensor.dtype),
            atol=1e-4)
        tf.debugging.assert_near(
            model_obj.observation_noise_fn.covariance(), tf.convert_to_tensor(
                obs_mtx_noise @ np.transpose(obs_mtx_noise), dtype=dtype_tensor.dtype),
            atol=1e-4)

        # Assert plusnoise_fn
        tf.debugging.assert_equal(
            model_obj.observation_plusnoise_fn(test_state).mean(),
            tf.convert_to_tensor(model_obj.observation_fn(test_state), dtype=dtype_tensor.dtype),
        )
        tf.debugging.assert_equal(
            model_obj.transition_plusnoise_fn(test_state).mean(),
            tf.convert_to_tensor(model_obj.transition_fn(test_state), dtype=dtype_tensor.dtype),
        )

        tf.debugging.assert_near(
            model_obj.transition_plusnoise_fn(test_state).covariance(), tf.convert_to_tensor(
                state_mtx_noise @ np.transpose(state_mtx_noise),
                dtype=dtype_tensor.dtype),
            atol=1e-4)
        tf.debugging.assert_near(
            model_obj.observation_plusnoise_fn(test_state).covariance(), tf.convert_to_tensor(
                obs_mtx_noise @ np.transpose(obs_mtx_noise), dtype=dtype_tensor.dtype),
            atol=1e-4)
