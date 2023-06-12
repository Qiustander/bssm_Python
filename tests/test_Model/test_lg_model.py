import pytest
import tensorflow as tf
import  numpy as np
from Models.ssm_lg import LinearGaussianSSM


class TestLinearGaussianModel:

    def test_univariate_model_shape(self):
        num_timesteps = 100
        state_dim = 1
        observed_dim = 1
        prior_mean = np.ones(shape=state_dim)
        prior_cov = np.ones(shape=[state_dim, state_dim])
        # input_state = np.random.randn(state_dim,)
        # input_obs = np.random.randn(observed_dim,)
        state_mtx = np.random.randn(state_dim, state_dim)
        obs_mtx = np.random.randn(observed_dim,)
        state_mtx_noise = np.random.randn(state_dim, state_dim)
        obs_mtx_noise = np.random.randn(observed_dim, observed_dim)

        model_obj = LinearGaussianSSM.create_model(num_timesteps=num_timesteps,
                                                   observation_size=observed_dim,
                                                   latent_size=state_dim,
                                                   initial_state_mean=prior_mean,
                                                   initial_state_cov=prior_cov,
                                                   state_noise_std=state_mtx_noise,
                                                   obs_noise_std=obs_mtx_noise,
                                                   obs_mtx=obs_mtx,
                                                   state_mtx=state_mtx)
        tf.debugging.assert_shapes([(model_obj.transition_matrix, (state_dim, state_dim)),
                                    (model_obj.transition_noise.sample(), (state_dim,)),
                                    (model_obj.observation_matrix, (observed_dim, state_dim)),
                                    (model_obj.observation_noise.sample(), (observed_dim,)),
                                    (model_obj.initial_state_prior.sample(), (state_dim,)),
                                        ])

    def test_multivariate_model_shape(self):
        num_timesteps = 100
        state_dim = 1
        observed_dim = 5
        prior_mean = np.ones(shape=state_dim)
        prior_cov = np.ones(shape=[state_dim, state_dim])
        # input_state = np.random.randn(state_dim,)
        # input_obs = np.random.randn(observed_dim,)
        state_mtx = np.random.randn(state_dim, state_dim)
        obs_mtx = np.random.randn(observed_dim, state_dim)
        state_mtx_noise = np.random.randn(state_dim, state_dim)
        obs_mtx_noise = np.random.randn(observed_dim, observed_dim)

        model_obj = LinearGaussianSSM.create_model(num_timesteps=num_timesteps,
                                                   observation_size=observed_dim,
                                                   latent_size=state_dim,
                                                   initial_state_mean=prior_mean,
                                                   initial_state_cov=prior_cov,
                                                   state_noise_std=state_mtx_noise,
                                                   obs_noise_std=obs_mtx_noise,
                                                   obs_mtx=obs_mtx,
                                                   state_mtx=state_mtx)
        tf.debugging.assert_shapes([(model_obj.transition_matrix, (state_dim, state_dim)),
                                    (model_obj.transition_noise.sample(), (state_dim,)),
                                    (model_obj.observation_matrix, (observed_dim, state_dim)),
                                    (model_obj.observation_noise.sample(), (observed_dim,)),
                                    (model_obj.initial_state_prior.sample(), (state_dim,)),
                                        ])

    def test_multivariate_multidim_model_shape(self):
        num_timesteps = 100
        state_dim = 4
        observed_dim = 5
        prior_mean = np.ones(shape=state_dim)
        prior_cov = np.ones(shape=[state_dim, state_dim])
        # input_state = np.random.randn(state_dim,)
        # input_obs = np.random.randn(observed_dim,)
        state_mtx = np.random.randn(state_dim, state_dim)
        obs_mtx = np.random.randn(observed_dim, state_dim)
        state_mtx_noise = np.random.randn(state_dim, state_dim)
        obs_mtx_noise = np.random.randn(observed_dim, observed_dim)

        model_obj = LinearGaussianSSM.create_model(num_timesteps=num_timesteps,
                                                   observation_size=observed_dim,
                                                   latent_size=state_dim,
                                                   initial_state_mean=prior_mean,
                                                   initial_state_cov=prior_cov,
                                                   state_noise_std=state_mtx_noise,
                                                   obs_noise_std=obs_mtx_noise,
                                                   obs_mtx=obs_mtx,
                                                   state_mtx=state_mtx)
        tf.debugging.assert_shapes([(model_obj.transition_matrix, (state_dim, state_dim)),
                                    (model_obj.transition_noise.sample(), (state_dim,)),
                                    (model_obj.observation_matrix, (observed_dim, state_dim)),
                                    (model_obj.observation_noise.sample(), (observed_dim,)),
                                    (model_obj.initial_state_prior.sample(), (state_dim,)),
                                        ])

    def test_univariate_multidim_model_shape(self):
        num_timesteps = 100
        state_dim = 4
        observed_dim = 1
        prior_mean = np.ones(shape=state_dim)
        prior_cov = np.ones(shape=[state_dim, state_dim])
        # input_state = np.random.randn(state_dim,)
        # input_obs = np.random.randn(observed_dim,)
        state_mtx = np.random.randn(state_dim, state_dim)
        obs_mtx = np.random.randn(observed_dim, state_dim)
        state_mtx_noise = np.random.randn(state_dim, state_dim)
        obs_mtx_noise = np.random.randn(observed_dim, observed_dim)

        model_obj = LinearGaussianSSM.create_model(num_timesteps=num_timesteps,
                                                   observation_size=observed_dim,
                                                   latent_size=state_dim,
                                                   initial_state_mean=prior_mean,
                                                   initial_state_cov=prior_cov,
                                                   state_noise_std=state_mtx_noise,
                                                   obs_noise_std=obs_mtx_noise,
                                                   obs_mtx=obs_mtx,
                                                   state_mtx=state_mtx)
        tf.debugging.assert_shapes([(model_obj.transition_matrix, (state_dim, state_dim)),
                                    (model_obj.transition_noise.sample(), (state_dim,)),
                                    (model_obj.observation_matrix, (observed_dim, state_dim)),
                                    (model_obj.observation_noise.sample(), (observed_dim,)),
                                    (model_obj.initial_state_prior.sample(), (state_dim,)),
                                        ])

    def test_multivariate_multidim_model_tv(self):
        num_timesteps = 20
        state_dim = 4
        observed_dim = 5
        prior_mean = np.ones(shape=state_dim)
        prior_cov = np.ones(shape=[state_dim, state_dim])
        input_state = np.random.randn(state_dim,)
        input_obs = np.random.randn(observed_dim,)
        state_mtx = np.random.randn(state_dim, state_dim, num_timesteps)
        obs_mtx = np.random.randn(observed_dim, state_dim, num_timesteps)
        state_mtx_noise = np.random.rand(state_dim, state_dim)
        obs_mtx_noise = np.random.rand(observed_dim, observed_dim)

        model_obj = LinearGaussianSSM.create_model(num_timesteps=num_timesteps,
                                                   observation_size=observed_dim,
                                                   latent_size=state_dim,
                                                   initial_state_mean=prior_mean,
                                                   initial_state_cov=prior_cov,
                                                   state_noise_std=state_mtx_noise,
                                                   obs_noise_std=obs_mtx_noise,
                                                   obs_mtx=obs_mtx,
                                                   state_mtx=state_mtx,
                                                   input_obs=input_obs,
                                                   input_state=input_state)
        for i in range(num_timesteps):
            tf.debugging.assert_shapes([(model_obj.transition_matrix(i).to_dense(), (state_dim, state_dim)),
                                        (model_obj.observation_matrix(i).to_dense(), (observed_dim, state_dim)),
                                        ])
        tf.debugging.assert_shapes([
                                    (model_obj.transition_noise.sample(), (state_dim,)),
                                    (model_obj.observation_noise.sample(), (observed_dim,)),
                                    (model_obj.initial_state_prior.sample(), (state_dim,)),
                                        ])

    def test_multivariate_multidim_model_noise_tv(self):
        num_timesteps = 20
        state_dim = 4
        observed_dim = 5
        prior_mean = np.ones(shape=state_dim)
        prior_cov = np.ones(shape=[state_dim, state_dim])
        input_state = np.random.randn(state_dim, )
        input_obs = np.random.randn(observed_dim, )
        state_mtx = np.random.randn(state_dim, state_dim)
        obs_mtx = np.random.randn(observed_dim, state_dim)
        state_mtx_noise = np.random.rand(state_dim, state_dim, num_timesteps)
        obs_mtx_noise = np.random.rand(observed_dim, observed_dim, num_timesteps)

        model_obj = LinearGaussianSSM.create_model(num_timesteps=num_timesteps,
                                                   observation_size=observed_dim,
                                                   latent_size=state_dim,
                                                   initial_state_mean=prior_mean,
                                                   initial_state_cov=prior_cov,
                                                   state_noise_std=state_mtx_noise,
                                                   obs_noise_std=obs_mtx_noise,
                                                   obs_mtx=obs_mtx,
                                                   state_mtx=state_mtx,
                                                   input_obs=input_obs,
                                                   input_state=input_state)
        for i in range(num_timesteps):
            tf.debugging.assert_shapes([(model_obj.transition_noise(i).sample(), (state_dim, )),
                                        (model_obj.observation_noise(i).sample(), (observed_dim, )),
                                        ])
            tf.debugging.assert_equal(
                model_obj.transition_noise(i).mean(), tf.convert_to_tensor(input_state, dtype=model_obj.transition_matrix.dtype),
            )
            tf.debugging.assert_equal(
                model_obj.observation_noise(i).mean(), tf.convert_to_tensor(input_obs, dtype=model_obj.transition_matrix.dtype),
            )
            print(f"{i}-th pass!")
            tf.debugging.assert_near(
                model_obj.transition_noise(i).covariance(), tf.convert_to_tensor(
                    state_mtx_noise[..., i] @ np.transpose(state_mtx_noise[..., i]), dtype=model_obj.transition_matrix.dtype),
            atol=1e-4)
            tf.debugging.assert_near(
                model_obj.observation_noise(i).covariance(), tf.convert_to_tensor(
                    obs_mtx_noise[..., i] @ np.transpose(obs_mtx_noise[..., i]), dtype=model_obj.transition_matrix.dtype),
            atol=1e-4)

        tf.debugging.assert_shapes([(model_obj.transition_matrix, (state_dim, state_dim)),
                                    (model_obj.observation_matrix, (observed_dim, state_dim)),
                                    (model_obj.initial_state_prior.sample(), (state_dim,)),
                                        ])

    def test_multivariate_multidim_model_input_tv(self):
        num_timesteps = 20
        state_dim = 4
        observed_dim = 5
        prior_mean = np.ones(shape=state_dim)
        prior_cov = np.ones(shape=[state_dim, state_dim])
        input_state = np.random.randn(state_dim, num_timesteps)
        input_obs = np.random.randn(observed_dim, num_timesteps)
        state_mtx = np.random.randn(state_dim, state_dim)
        obs_mtx = np.random.randn(observed_dim, state_dim)
        state_mtx_noise = np.random.rand(state_dim, state_dim)
        obs_mtx_noise = np.random.rand(observed_dim, observed_dim)

        model_obj = LinearGaussianSSM.create_model(num_timesteps=num_timesteps,
                                                   observation_size=observed_dim,
                                                   latent_size=state_dim,
                                                   initial_state_mean=prior_mean,
                                                   initial_state_cov=prior_cov,
                                                   state_noise_std=state_mtx_noise,
                                                   obs_noise_std=obs_mtx_noise,
                                                   obs_mtx=obs_mtx,
                                                   state_mtx=state_mtx,
                                                   input_obs=input_obs,
                                                   input_state=input_state)
        for i in range(num_timesteps):
            tf.debugging.assert_shapes([(model_obj.transition_noise(i).sample(), (state_dim,)),
                                        (model_obj.observation_noise(i).sample(), (observed_dim,)),
                                        ])
            tf.debugging.assert_equal(
                model_obj.transition_noise(i).mean(),
                tf.convert_to_tensor(input_state[..., i], dtype=model_obj.transition_matrix.dtype),
            )
            tf.debugging.assert_equal(
                model_obj.observation_noise(i).mean(),
                tf.convert_to_tensor(input_obs[..., i], dtype=model_obj.transition_matrix.dtype),
            )

            tf.debugging.assert_near(
                model_obj.transition_noise(i).covariance(), tf.convert_to_tensor(
                    state_mtx_noise @ np.transpose(state_mtx_noise),
                    dtype=model_obj.transition_matrix.dtype),
                atol=1e-4)
            tf.debugging.assert_near(
                model_obj.observation_noise(i).covariance(), tf.convert_to_tensor(
                    obs_mtx_noise @ np.transpose(obs_mtx_noise),
                    dtype=model_obj.transition_matrix.dtype),
                atol=1e-4)

        tf.debugging.assert_shapes([(model_obj.transition_matrix, (state_dim, state_dim)),
                                    (model_obj.observation_matrix, (observed_dim, state_dim)),
                                    (model_obj.initial_state_prior.sample(), (state_dim,)),
                                    ])