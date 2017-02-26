# Author: Alber Erre - Feb 2017 
# Github: https://github.com/AlberErre
"""
Made using python 3 :D

Feel free to change your dataset, this code was developed for 2D data only

"""

import tensorflow as tf
import numpy as np

## Dataset, just 2D data created randomly
_realWeights =  0.6
_realBiases =   0.4
x_data = np.random.rand(50).astype(np.float32)
y_data = _realWeights*x_data + _realBiases 

## Tensorflow variables (update these until obtain _realWeights and _realBiases)
_modelWeights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
_modelBiases = tf.Variable(tf.zeros([1]))

## Model
Predicted_y = _modelWeights*x_data + _modelBiases

## Calculate de square error between predicted and real "y" values
Loss_error = tf.reduce_mean(tf.square(Predicted_y - y_data))

## update variables using gradient descent and minimize Loss_error
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

## Run the model and init
# tf.initialize_all_variables(), no longer available after 2017-03-02. 
init = tf.global_variables_initializer()
sess = tf.Session()



