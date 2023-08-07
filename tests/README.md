## Unit Test and Integration Test of TFP Based BSSM

All testing are done by `pytest` and `tf.testing`
### [test_numerical_case](test_with_bssm)
Contains the integration test and the comparsion with orignal `bssm` package.


### [test_Inference](test_Inference)
Test for all Inference Methods
* Kalman-Based Methods [test_Kalman](test_Inference%2Ftest_Kalman):
    * test extended Kalman filter [test_ekf.py](test_Inference%2Ftest_Kalman%2Ftest_ekf.py)
    * test extended Kalman smoother [test_eksmoother.py](test_Inference%2Ftest_Kalman%2Ftest_eksmoother.py)
    * test ensemble Kalman filter [test_enkf.py](test_Inference%2Ftest_Kalman%2Ftest_enkf.py)
    * test unscented Kalman filter [test_ukf.py](test_Inference%2Ftest_Kalman%2Ftest_ukf.py)
* Sequential Monte Carlo


### [test_Model](test_Model)
Test the SSM model class
* test the input arguments for linear Guassian models [test_check_argument.py](test_Model%2Ftest_check_argument.py)
* test the linear Gaussian model class
* test the nonlinear Gaussian model class
