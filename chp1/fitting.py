import sys, os
sys.path.append("/data/yuming/eeg-dnn")
import utils
import numpy as np
import tensorflow as tf
import sonnet as snt
import matplotlib.pyplot as plt
import pdb

utils.GPUConfig()(gpu = 1)

class GenSample(snt.Module):
    def __init__(self, n_samples = 32, name = "gen_sample"):
        super(GenSample, self).__init__(name = name)
        self._n_samples = n_samples
        self._true_fun = lambda X: tf.math.cos(1.5 * np.pi * X)

    def __call__(self):
        X = tf.sort(tf.random.uniform([self._n_samples, 1]))
        y = self._true_fun(X) + tf.random.normal([self._n_samples, 1]) * 0.1

        return (X, y)

    @property
    def true_fun(self):
        return self._true_fun

log = utils.Log('output')
X, y = GenSample()()

degrees = [1, 4, 15]

num_epochs = 10000
opt = snt.optimizers.SGD(learning_rate = 0.05)

X_test = tf.linspace([0.0], [1.0], 100, axis = 0)
fig = plt.figure(figsize= (12.8, 4.8))

pt = 10
plt.rc('font', size=pt)          # controls default text sizes
plt.rc('axes', titlesize=pt)     # fontsize of the axes title
plt.rc('axes', labelsize=pt)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=pt)    # fontsize of the tick labels
plt.rc('ytick', labelsize=pt)    # fontsize of the tick labels
plt.rc('legend', fontsize=pt)    # legend fontsize

true_fun = lambda X: tf.math.cos(1.5 * np.pi * X)

for i in range(len(degrees)):
    model = utils.PolynomialRegression(degree=degrees[i])

    # if i == 1: pdb.set_trace() 

    for _ in range(num_epochs):
        with tf.GradientTape(persistent = True) as tape:
            logits = model(X)
            loss = utils.LossRegression()(logits, y)

        params = model.trainable_variables
        grads = tape.gradient(loss, params)
        opt.apply(grads, params)

    del tape

    print("\ndegree: {}, loss: {}".format(degrees[i], loss.numpy()))

    ax = fig.add_subplot(1, len(degrees), i+1)
    y_poly_pred = model(X_test, is_training = False)    

    plt.plot(X_test, y_poly_pred, label="Model")
    plt.plot(X_test, true_fun(X_test), label="True function")
    plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
    plt.legend(loc="best")

fig.tight_layout()

plt2img = utils.Plot2Image()
log('image', 'samples', plt2img(plt, fig))


