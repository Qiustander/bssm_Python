import pytest
import numpy as np
import os.path as pth
import os

from Models.check_argument import *


class TestObservationTimeSeries(object):
    """
    Test observation time series y
    """

    def test_check_y_scalar_input(self):
        with pytest.raises(TypeError):
            check_y(10)  # Scalar input should raise TypeError

    def test_check_y_array_input(self):
        x = np.array([1, 2, 3])
        shape, result = check_y(x)
        assert shape == (3, 1)  # Expected shape is (3, 1)
        assert np.array_equal(result, x.reshape(-1, 1))  # Result should match the input array

    def test_check_y_2d_array_input(self):
        x = np.random.randn(2,3)
        shape, result = check_y(x)
        assert shape == (2, 3)  # Expected shape is (2, 3)
        assert np.array_equal(result, x)  # Result should match the input array

    def test_check_y_single_column_input(self):
        x = np.random.randn(3,)
        shape, result = check_y(x)
        assert shape == (3, 1)  # Expected shape is (3,)
        assert np.array_equal(result, x[..., None])  # Result should match the flattened input array

    def test_check_y_inf_values(self):
        x = np.array([[1, 2, 3], [4, np.inf, 6]])
        with pytest.raises(TypeError):
            check_y(x)  # Array with infinite values should raise TypeError

    def test_check_y_missing_values(self):
        x = np.array([[1, np.nan, 3], [4, 5, np.nan]])
        shape, result = check_y(x)
        assert shape == (2, 3)  # Expected shape is (2, 3)
        assert np.array_equal(result,
                              np.nan_to_num(x))  # Result should match the input array with an additional dimension

    def test_check_y_empty_array(self):
        with pytest.raises(TypeError):
            check_y(np.array([]))  # Empty array should raise ValueError


class TestRho(object):
    """
    Test rho parameter for the AR(1) model
    """

    def test_check_rho_scalar_input(self):
        x = 0.5
        result = check_rho(x)
        assert np.array_equal(result, np.array([x]))  # Result should be a 1D numpy array containing the scalar input

    def test_check_rho_list_input(self):
        x = [0.2]
        result = check_rho(x)
        assert np.array_equal(result, np.array(x))  # Result should be a 1D numpy array containing the input list

    def test_check_rho_tuple_input(self):
        x = (0.2)
        result = check_rho(x)
        assert np.array_equal(result, np.array(x)[None])  # Result should be a 1D numpy array containing the input list

    def test_check_rho_ndarray_input(self):
        x = np.array([0.3])
        result = check_rho(x)
        assert np.array_equal(result, x)  # Result should be the same as the input ndarray

    def test_check_rho_invalid_length(self):
        x = [0.4, 0.5]
        with pytest.raises(ValueError):
            check_rho(x)  # Input list with length > 1 should raise ValueError

    def test_check_rho_invalid_range(self):
        x = 1.2
        with pytest.raises(ValueError):
            check_rho(x)  # Input outside the range [-1, 1] should raise ValueError

    def test_check_rho_none(self):
        x = None
        with pytest.raises(ValueError):
            check_rho(x)  # Input with no valid return should raise ValueError


class TestInputObservation:
    def test_check_input_obs_none_input(self):
        p = 3
        n = 5
        result = check_input_obs(None, p, n)
        assert np.array_equal(result, np.zeros((p,)))  # None input should result in an array of zeros

    def test_check_input_obs_scalar_input(self):
        x = 10
        p = 1
        n = 5
        result = check_input_obs(x, p, n)
        assert np.array_equal(result, np.array([x]))  # Scalar input should result in a 1D numpy array

    def test_check_input_obs_univariate_scalar(self):
        x = np.array(5)
        p = 1
        n = 1
        result = check_input_obs(x, p, n)
        assert np.array_equal(result, np.array([x]))  # Univariate scalar input should result in a 1D numpy array

    def test_check_input_obs_univariate_array(self):
        p = 1
        n = 5
        x = np.random.randn(n,)
        result = check_input_obs(x, p, n)
        assert np.array_equal(result, x)  # Univariate array input should remain the same

    def test_check_input_obs_multivariate_matrix(self):
        p = 2
        n = 5
        x = np.random.randn(p,n)
        result = check_input_obs(x, p, n)
        assert np.array_equal(result, x)  # Multivariate matrix input should remain the same

    def test_check_univariate_scalar(self):
        x = np.array(5)
        p = 1
        n = 3
        assert np.array_equal(check_input_obs(x, p, n), x[None])  # Scalar input should result in a 1D numpy array

    def test_check_input_obs_invalid_univariate_array(self):
        x = np.array([1, 2, 3, 4, 5])
        p = 1
        n = 3
        with pytest.raises(ValueError):
            check_input_obs(x, p, n)  # Univariate array input with invalid length should raise ValueError

    def test_check_input_obs_invalid_multivariate_matrix(self):
        x = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        p = 3
        n = 5
        with pytest.raises(ValueError):
            check_input_obs(x, p, n)  # Multivariate matrix input with invalid number of series should raise ValueError

    def test_check_input_obs_invalid_multivariate_matrix_n(self):
        x = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        p = 3
        n = 7
        with pytest.raises(ValueError):
            check_input_obs(x, p,
                            n)  # Multivariate matrix input with invalid number of series should raise ValueError

    def test_check_input_obs_multivariate_matrix_constant(self):
        x = np.array([[1], [3], [5]])
        p = 3
        n = 5
        assert np.array_equal(check_input_obs(x, p, n), x.squeeze())  # Scalar input should result in a 1D numpy array


class TestInputState():
    def test_check_input_state_none_input(self):
        x = None
        m = 3
        n = 2
        expected_output = np.zeros((m,))
        assert np.array_equal(check_input_state(x, m, n), expected_output)

    def test_check_input_state_numeric_input(self):
        x = [1, 2, 3]
        m = 3
        n = 1
        expected_output = np.array(x)
        assert np.array_equal(check_input_state(x, m, n), expected_output)

    def test_check_input_state_invalid_type(self):
        x = "not_numeric"
        m = 2
        n = 1
        with pytest.raises(ValueError):
            check_input_state(x, m, n)

    def test_check_input_state_invalid_dimension(self):
        x = np.array([1, 2])
        m = 3
        n = 1
        with pytest.raises(ValueError):
            check_input_state(x, m, n)

    def test_check_input_state_valid_dimension(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])
        m = 3
        n = 2
        expected_output = x
        assert np.array_equal(check_input_state(x, m, n), expected_output)


class TestObservationNoise():

    def test_check_obs_mtx_noise_numeric_input(self):
        x = 0.5
        p = 1
        n = 10
        expected_output = np.array([[0.5]])
        assert np.array_equal(check_obs_mtx_noise(x, p, n), expected_output)

    def test_check_obs_mtx_noise_invalid_type(self):
        x = "not_numeric"
        p = 2
        n = 10
        with pytest.raises(ValueError):
            check_obs_mtx_noise(x, p, n)

    def test_check_obs_mtx_noise_univariate_scalar(self):
        x = np.array([0.2, 0.3])
        p = 1
        n = 2
        expected_output = x[None][None]
        assert np.array_equal(check_obs_mtx_noise(x, p, n), expected_output)

    def test_check_obs_mtx_noise_invalid_covariance_size(self):
        x = np.array([[1.0, 0.5], [0.5, 1.0], [0.5, 0.5]])
        p = 2
        n = 3
        with pytest.raises(ValueError):
            check_obs_mtx_noise(x, p, n)

    def test_check_obs_mtx_noise_valid_covariance_size(self):
        x = np.array([[1.0, 0.5], [0.5, 1.0]])
        p = 2
        n = 3
        expected_output = x
        assert np.array_equal(check_obs_mtx_noise(x, p, n), expected_output)

    def test_check_obs_mtx_noise_invalid_covariance_shape(self):
        x = np.array([[[1.0, 0.5], [0.5, 1.0]], [[1.0, 0.5], [0.5, 1.0]]])
        p = 2
        n = 3
        with pytest.raises(ValueError):
            check_obs_mtx_noise(x, p, n)

    def test_check_obs_mtx_noise_valid_covariance_shape(self):
        x = np.array([[[1., 1., 1.],
                       [0.5, 0.5, 0.5]],
                      [[0.5, 0.5, 0.5],
                       [1., 1., 1.]]])
        p = 2
        n = 3
        expected_output = x
        assert np.array_equal(check_obs_mtx_noise(x, p, n), expected_output)


class TestStateNoise():

    def test_check_state_noise_numeric_input(self):
        x = 0.5
        m = 1
        n = 10
        expected_output = np.array([[0.5]])
        assert np.array_equal(check_state_noise(x, m, n), expected_output)

    def test_check_state_noise_invalid_type(self):
        x = "not_numeric"
        m = 2
        n = 10
        with pytest.raises(ValueError):
            check_state_noise(x, m, n)

    def test_check_state_noise_scalar_covariance(self):
        x = np.array(0.2)
        m = 2
        n = 10
        with pytest.raises(ValueError):
            check_state_noise(x, m, n)

    def test_check_state_noise_invalid_covariance_shape(self):
        x = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2]])
        m = 2
        n = 10
        with pytest.raises(ValueError):
            check_state_noise(x, m, n)

    def test_check_state_noise_valid_covariance_shape(self):
        x = np.array([[1.0, 0.5], [0.5, 1.0]])
        m = 2
        n = 10
        expected_output = x
        assert np.array_equal(check_state_noise(x, m, n), expected_output)

    def test_check_state_noise_invalid_covariance_array_shape(self):
        x = np.array([[[1.0, 0.5], [0.5, 1.0]], [[1.0, 0.5], [0.5, 1.0]], [[1.0, 0.5], [0.5, 1.0]]])
        m = 2
        n = 10
        with pytest.raises(ValueError):
            check_state_noise(x, m, n)

    def test_check_state_noise_valid_covariance_array_shape(self):
        x = np.array([[[1.0, 0.5], [0.5, 1.0]], [[1.0, 0.5], [0.5, 1.0]]]).repeat(5, axis=-1)
        m = 2
        n = 10
        expected_output = x
        assert np.array_equal(check_state_noise(x, m, n), expected_output)

    def test_check_state_noise_valid_droprank_array_shape(self):
        m = 5
        n = 10
        x = np.random.randn(m, m-2)
        expected_output = x
        assert np.array_equal(check_state_noise(x, m, n), expected_output)

    def test_check_state_noise_invalid_droprank_array_shape(self):
        m = 5
        n = 10
        x = np.random.randn(m, m+2)
        with pytest.raises(ValueError):
            check_state_noise(x, m, n)

class TestObsMatrix:

    def test_check_obs_mtx_scalar_input_univariate(self):
        x = 0.5
        p = 1
        n = 10
        m = 1
        expected_output = np.array([[x]])
        assert np.array_equal(check_obs_mtx(x, p, n, m), expected_output)
        assert np.array_equal(check_obs_mtx([x], p, n, m), expected_output)

    def test_check_obs_mtx_scalar_input_multivariate(self):
        x = 0.5
        p = 2
        n = 10
        m = 1
        with pytest.raises(ValueError):
            check_obs_mtx(x, p, n, m)

    def test_check_obs_mtx_invalid_type(self):
        x = "not_numeric"
        p = 2
        n = 10
        m = 1
        with pytest.raises(TypeError):
            check_obs_mtx(x, p, n, m)

    def test_check_obs_mtx_invalid_shape_univariate(self):
        p = 1
        n = 10
        m = 2
        x = np.random.rand(p+1, m)
        with pytest.raises(ValueError):
            check_obs_mtx(x, p, n, m)

    def test_check_obs_mtx_valid_shape_univariate_fixed(self):
        p = 1
        n = 10
        m = 4
        x = np.random.rand(p, m)
        expected_output = x
        assert np.array_equal(check_obs_mtx(x, p, n, m), expected_output)

    def test_check_obs_mtx_valid_shape_univariate_time_varying(self):
        p = 1
        n = 10
        m = 2
        x = np.random.rand(p, m, n)
        expected_output = x
        assert np.array_equal(check_obs_mtx(x, p, n, m), expected_output)

    def test_check_obs_mtx_invalid_shape_multivariate(self):
        p = 2
        n = 10
        m = 2
        x = np.random.rand(p, m, n+1)
        with pytest.raises(ValueError):
            check_obs_mtx(x, p, n, m)

    def test_check_obs_mtx_valid_shape_multivariate_time_varying(self):
        p = 3
        n = 7
        m = 2
        x = np.random.rand(p, m, n)
        expected_output = x
        assert np.array_equal(check_obs_mtx(x, p, n, m), expected_output)

    def test_check_obs_mtx_valid_shape_multivariate_fixed(self):
        p = 3
        n = 7
        m = 1
        x = np.random.rand(p, m)
        expected_output = x
        assert np.array_equal(check_obs_mtx(x, p, n, m), expected_output)


class TestStateMatrix:
    def test_check_state_mtx_scalar_input_univariate(self):
        x = 0.5
        m = 1
        n = 10
        expected_output = np.array([[0.5]])
        assert np.array_equal(check_state_mtx(x, m, n), expected_output)

    def test_check_state_mtx_scalar_input_multivariate(self):
        x = 0.5
        m = 2
        n = 10
        with pytest.raises(ValueError):
            check_state_mtx(x, m, n)

    def test_check_state_mtx_invalid_type(self):
        x = "not_numeric"
        m = 2
        n = 10
        with pytest.raises(TypeError):
            check_state_mtx(x, m, n)

    def test_check_state_mtx_invalid_shape_univariate(self):
        m = 2
        n = 10
        x = np.random.rand(m,)
        with pytest.raises(ValueError):
            check_state_mtx(x, m, n)

    def test_check_state_mtx_valid_shape_univariate(self):
        x = np.array([[1.0, 0.5], [0.5, 1.0]])
        m = 2
        n = 10
        expected_output = x
        assert np.array_equal(check_state_mtx(x, m, n), expected_output)

    def test_check_state_mtx_invalid_shape_multivariate(self):
        x = np.array([[[1.0, 0.5], [0.5, 1.0]], [[1.0, 0.5], [0.5, 1.0]], [[1.0, 0.5], [0.5, 1.0]]])
        m = 2
        n = 10
        with pytest.raises(ValueError):
            check_state_mtx(x, m, n)

    def test_check_state_mtx_valid_shape_multivariate(self):
        m = 2
        n = 10
        x = np.random.rand(m, m+1)
        with pytest.raises(ValueError):
            check_state_mtx(x, m, n)

    def test_check_state_mtx_valid_shape_multivariate_squeeze(self):
        m = 2
        n = 10
        x = np.random.rand(m,m,1)
        expected_output = x
        assert np.array_equal(check_state_mtx(x, m, n), expected_output.squeeze())

    def test_check_state_mtx_valid_shape_multivariate_time_varying(self):
        m = 2
        n = 10
        x = np.random.rand(m,m,n)
        expected_output = x
        assert np.array_equal(check_state_mtx(x, m, n), expected_output)

    def test_check_state_mtx_valid_shape_univariate_time_varying(self):
        m = 1
        n = 10
        x = np.random.rand(m, m, n)
        expected_output = x
        assert np.array_equal(check_state_mtx(x, m, n), expected_output)

    def test_check_state_mtx_valid_shape_univariate_time_varying_boardcast(self):
        m = 1
        n = 10
        x = np.random.rand(m, n)
        expected_output = x
        assert np.array_equal(check_state_mtx(x, m, n), expected_output[None])


class TestPriorMean():

    def test_check_prior_mean_none_input(self):
        x = None
        m = 3
        expected_output = np.zeros(m)
        assert np.array_equal(check_prior_mean(x, m), expected_output)

    def test_check_prior_mean_scalar_input(self):
        x = 0.5
        m = 3
        expected_output = np.array([0.5, 0.5, 0.5])
        assert np.array_equal(check_prior_mean(x, m), expected_output)

    def test_check_prior_mean_invalid_type(self):
        x = "not_numeric"
        m = 3
        with pytest.raises(TypeError):
            check_prior_mean(x, m)

    def test_check_prior_mean_invalid_shape(self):
        x = np.array([1.0, 0.5])
        m = 3
        with pytest.raises(ValueError):
            check_prior_mean(x, m)

    def test_check_prior_mean_valid_shape(self):
        x = np.array([1.0, 0.5, 0.3])
        m = 3
        expected_output = x
        assert np.array_equal(check_prior_mean(x, m), expected_output)


class TestPriorCov():

    def test_check_prior_cov_none_input(self):
        x = None
        m = 3
        expected_output = np.zeros((m, m))
        assert np.array_equal(check_prior_cov(x, m), expected_output)

    def test_check_prior_cov_scalar_input(self):
        x = 0.5
        m = 1
        expected_output = np.array([[0.5]])
        assert np.array_equal(check_prior_cov(x, m), expected_output)


    def test_check_prior_cov_list_input(self):
        x = [0.5]
        m = 1
        expected_output = np.array([[0.5]])
        assert np.array_equal(check_prior_cov(x, m), expected_output)

    def test_check_invalid_prior_cov_scalar_input(self):
        x = 0.5
        m = 3
        with pytest.raises(ValueError):
            check_prior_cov(x, m)

    def test_check_prior_cov_invalid_type(self):
        x = "not_numeric"
        m = 3
        with pytest.raises(TypeError):
            check_prior_cov(x, m)

    def test_check_prior_cov_invalid_shape(self):
        m = 3
        x = np.random.rand(m,)
        with pytest.raises(ValueError):
            check_prior_cov(x, m)

    def test_check_prior_cov_valid_shape(self):
        m = 3
        x = np.random.rand(m, m)
        assert np.array_equal(check_prior_cov(x, m), x)



