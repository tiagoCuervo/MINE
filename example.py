import numpy as np
import matplotlib.pyplot as plt
from mine import MINE
from sklearn.feature_selection import mutual_info_regression

# Generates data by sampling from two correlated Gaussian variables
dim = 1
variance = 0.2
sampleSize = 2000

xSamples = np.sign(np.random.normal(0., 1., [sampleSize, dim]))
ySamples = xSamples + np.random.normal(0., np.sqrt(variance), [sampleSize, dim])
pyx = np.exp(-(ySamples - xSamples) ** 2 / (2 * variance))
pyxMinus = np.exp(-(ySamples + 1) ** 2 / (2 * variance))
pyxPlus = np.exp(-(ySamples - 1) ** 2 / (2 * variance))

mi = np.average(np.log(pyx / (0.5 * pyxMinus + 0.5 * pyxPlus)))

miEstimator = MINE(dim, archSpecs={
    'layerSizes': [32] * 1,
    'activationFunctions': ['relu'] * 1
}, divergenceMeasure='KL', learningRate=1e-3)


ySamplesMarginal = np.random.permutation(ySamples)

# noinspection PyUnresolvedReferences
estimatedMI, estimationHistory = miEstimator.calcMI(xSamples, ySamples, xSamples, ySamplesMarginal,
                                                    batchSize=sampleSize, numEpochs=2000)

print("Real MI: {}, estimated MI: {}".format(mi, estimatedMI))
print("Estimated MI: {}".format(estimatedMI))
epochs = np.arange(len(estimationHistory))
plt.plot(epochs, estimationHistory)
plt.plot(epochs, mi * np.ones(len(estimationHistory)))
plt.xlabel('Epochs')
plt.ylabel('Estimated MI')
plt.legend(['Estimated', 'Real'])
plt.show()
