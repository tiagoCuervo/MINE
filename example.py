import numpy as np
import matplotlib.pyplot as plt
from mine import MINE

# Generates data by sampling from two correlated Gaussian variables
dim = 1
variance = 0.2
sampleSize = 1000

xSamples = np.sign(np.random.normal(0., 1., [sampleSize, dim]))
ySamples = xSamples + np.random.normal(0., np.sqrt(variance), [sampleSize, dim])
pyx = np.exp(-(ySamples - xSamples) ** 2 / (2 * variance))
pyxMinus = np.exp(-(ySamples + 1) ** 2 / (2 * variance))
pyxPlus = np.exp(-(ySamples - 1) ** 2 / (2 * variance))

mi = np.average(np.log(pyx / (0.5 * pyxMinus + 0.5 * pyxPlus)))

miEstimator = MINE(1, archSpecs={
    'layerSizes': [32],
    'useBias': True,
    'activationFunctions': ['relu']
})


ySamplesMarginal = np.random.permutation(ySamples)

# noinspection PyUnresolvedReferences
estimatedMI, estimationHistory = miEstimator.calcMI(xSamples, ySamples, xSamples, ySamplesMarginal)

print("Real MI: {}, estimated MI: {}".format(mi, estimatedMI))
epochs = np.arange(len(estimationHistory))
plt.plot(epochs, estimationHistory)
plt.plot(epochs, mi * np.ones(len(estimationHistory)))
plt.xlabel('Epochs')
plt.ylabel('Estimated MI')
plt.legend(['Estimated', 'Real'])
plt.show()