This is the respository for Python version of the bssm package, armed with Tensorflow and Tensorflow Probabiliaty.

Dependency: rpy2 -  not on Windows

### Current Issues:
* modify the UKF and recompile the bssm package 
* Add the unittest of the Kalman-Based Methods
* rewrite the extended kalman smoother
* Rewrite the testing into time-varying type, and EKF
* add likelihood of UKF, and rewrite UKF as batch based
* Deal with the drop-rank state noise matrix R_t
* Check why particle filter fails for multi-variate constant dynamics