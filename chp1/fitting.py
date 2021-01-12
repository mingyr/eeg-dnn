import sys, os
sys.path.append("/data/yuming/eeg-dnn")
import utils
import numpy as np
import tensorflow as tf
import sonnet as snt
import matplotlib.pyplot as plt
import pdb

utils.GPUConfig()(gpu = 1)


class PolynomialFn(snt.Module):
    def __init__(self, name = "polynomial_fn"):
        super(PolynomialFn, self).__init__(name = name)

    @snt.once
    def _initialize(self):
        self._coeffs = [tf.constant(c, dtype = tf.float32) for c in [-6.4, 6.4, 1.6]]
        self._ini = tf.constant([0.0], dtype = tf.float32)

    def __call__(self, x):
        self._initialize()
        y = tf.scan(lambda unused, v: tf.math.polyval(self._coeffs, v), x, self._ini)

        return y
    

true_fun = PolynomialFn() # lambda X: tf.math.cos(1.5 * np.pi * X)

class GenSample(snt.Module):
    def __init__(self, n_samples = 8, name = "gen_sample"):
        super(GenSample, self).__init__(name = name)
        self._n_samples = n_samples
        # self._true_fun = lambda X: tf.math.cos(1.5 * np.pi * X)
        self._true_fun = true_fun

    def __call__(self):
        X = tf.sort(tf.linspace([0.0], [1.0], 8, axis = 0))
        y = self._true_fun(X) + tf.random.normal([self._n_samples, 1]) * 0.1

        return (X, y)

    @property
    def true_fun(self):
        return self._true_fun

log = utils.Log('output')
X, y = GenSample()()

degrees = [1, 2, 10]

num_epochs = 100000
opt = snt.optimizers.SGD(learning_rate = 0.2)

X_test = tf.linspace([0.0], [1.0], 100, axis = 0)
fig = plt.figure(figsize= (12.8, 4.8))

pt = 11
plt.rc('font', size=pt)          # controls default text sizes
plt.rc('axes', titlesize=pt)     # fontsize of the axes title
plt.rc('axes', labelsize=pt)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=pt)    # fontsize of the tick labels
plt.rc('ytick', labelsize=pt)    # fontsize of the tick labels
plt.rc('legend', fontsize=pt)    # legend fontsize

# true_fun = lambda X: tf.math.cos(1.5 * np.pi * X)

for i in range(len(degrees)):
    model = utils.PolynomialRegression(degree=degrees[i])

    # if i == 1: pdb.set_trace() 

    for _ in range(num_epochs):
        with tf.GradientTape(persistent = True) as tape:
            logits = model(X)
            loss = utils.LossRegression()(logits, y)

            # print("loss: {}".format(loss))  
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
    plt.xlim((0.0, 1.0))
    plt.ylim((0.0, 4.0))
    plt.legend(loc="best")

fig.tight_layout()

plt2img = utils.Plot2Image()
log('image', 'samples', plt2img(plt, fig))


