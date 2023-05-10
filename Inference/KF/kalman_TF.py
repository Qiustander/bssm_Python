# Python specific imports that will make our job easier and our code prettier
from collections import namedtuple
from functools import partial
import math
import time
from tqdm.auto import trange, tqdm

# TF specific imports that we will use to code the logic
from tensorflow import function
import tensorflow as tf
import tensorflow_probability as tfp

# Auxiliary libraries that we will use to report results and create the data
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

mm = tf.linalg.matmul
mv = tf.linalg.matvec

"""
Implement the Kalman filter via Tensorflow Probability

"""

