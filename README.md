This is the respository for Python version of the bssm package, armed with Tensorflow and Tensorflow Probabiliaty.

Dependency: rpy2 -  not on Windows

### Current Issues:
* modify the UKF and recompile the bssm package 
* Deal with the drop-rank state noise matrix R_t
* update model in the tf static graph, need to rewrite check_argument.py to be compatible with tf.Tensor
