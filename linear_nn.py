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

## Training: update variables using gradient descent and minimize Loss_error
optimizer = tf.train.GradientDescentOptimizer(0.5)
trainModel = optimizer.minimize(Loss_error)

## Run the model and init
# tf.initialize_all_variables(), no longer available after 2017-03-02. 
init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

# Increasing Range will increase the accuracy of the model
for step in range(116):
    sess.run(trainModel)
    if step % 15 == 0: # Batch: 15, plot results every 15 observations
        print("Observation:", step, 
              "Model Weights", sess.run(_modelWeights),
              "Model Biases:", sess.run(_modelBiases),
              "Real Weights:", _realWeights, 
              "Real Biases:", _realBiases)

## Both _modelWeights and _modelBiases approximate to real values in every iteration :D It works!



################################################
## USING NOISE:

noise = np.random.normal(0, 0.06, x_data.shape)
y_data_noisy = _realWeights*(x_data + noise) + _realBiases # Y values following a linear structure

plt.scatter(x_data,y_data_noisy)
plt.show()

#####
HIGH_noise = np.random.rand(100).astype(np.float32)
y_data_HIGH_noise = _realWeights*(x_data + HIGH_noise) + _realBiases # Y values following a linear structure

plt.scatter(x_data,y_data_HIGH_noise)
plt.show()

####
Loss_error_noisy_HIGH = tf.reduce_mean(tf.square(Predicted_y - y_data_noisy)) 
    # We can use "y_data_HIGH_noise" instead of "y_data_noisy"
    # Depends of how much noise you want :D
optimizer = tf.train.GradientDescentOptimizer(0.5)
			## We also can use "AdamOptimizer"

trainModel_noisy = optimizer.minimize(Loss_error_noisy_HIGH)

## Run the model and init (HIGH noise)
init = tf.initialize_all_variables()#, no longer availability after 2017-03-02. 
#init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Increasing Range will increase the accuracy of the model
for step in range(116):
    sess.run(trainModel_noisy)
    if step % 15 == 0: # Batch: 15, plot results every 15 observations
        print("Observation:", step, 
              "Model Weights", sess.run(_modelWeights),
              "Model Biases:", sess.run(_modelBiases),
              "Real Weights:", _realWeights, 
              "Real Biases:", _realBiases)

