import sys, os
sys.path.append("/data/yuming/eeg-dnn")
import utils
import numpy as np
import tensorflow as tf
import sonnet as snt
import matplotlib.pyplot as plt

utils.GPUConfig()(gpu = 1)

class GenSample(snt.Module):
    def __init__(self, n_samples = 32, name = "gen_sample"):
        super(GenSample, self).__init__(name = name)
        self._n_samples = n_samples
        self._true_fun = lambda X: tf.math.cos(1.5 * np.pi * X)

    def __call__(self):
        X = tf.sort(tf.random.uniform([self._n_samples]))
        y = self._true_fun(X) + tf.random.normal([self._n_samples]) * 0.1

        return (X, y)

log = utils.Log('output')
X, y = GenSample()()

print(X)
print(y)

figure = plt.figure(figsize=(10,10))
plt.plot(X, y, color = 'red', label="Actual")
plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="best")

plt2img = utils.Plot2Image()
log('image', 'samples', plt2img(plt, figure))



