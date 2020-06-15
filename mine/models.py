from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from mine.utils import logSumExp


# noinspection PyUnresolvedReferences,PyTypeChecker
class MLP(nn.Module):
    def __init__(self, inputSpaceDim, outputSpaceDim, archSpecs):
        super(MLP, self).__init__()
        layerSizes = [inputSpaceDim] + archSpecs['layerSizes'] + [outputSpaceDim]
        useBias = archSpecs['useBias']
        self.activationFunctions = archSpecs['activationFunctions']
        assert len(self.activationFunctions) == (len(layerSizes) - 1)
        self.layers = nn.ModuleList()
        for l in range(len(layerSizes) - 1):
            self.layers.append(nn.Linear(layerSizes[l], layerSizes[l + 1],
                                         bias=useBias if l < (len(layerSizes) - 2) else True))

    def forward(self, x):
        for l in range(len(self.layers)):
            activationFunction = self.activationFunctions[l]
            x = self.layers[l](x) if activationFunction.lower() == 'linear' \
                else eval('torch.' + activationFunction.lower())(self.layers[l](x))
        return x


# noinspection PyArgumentList,PyTypeChecker
class MINE(MLP):
    def __init__(self, inputSpaceDim, archSpecs, divergenceMeasure='KL', learningRate=0.01):
        archSpecs['activationFunctions'].append('linear')
        super().__init__(inputSpaceDim * 2, 1, archSpecs)
        self.divergenceMeasure = divergenceMeasure
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learningRate)

    def calcMI(self, xSamplesJoint, ySamplesJoint, xSamplesMarginal, ySamplesMarginal, numEpochs=500):
        numSamplesJoint = xSamplesJoint.shape[0]
        assert ySamplesJoint.shape[0] == numSamplesJoint
        assert ySamplesMarginal.shape[0] == xSamplesMarginal.shape[0]

        samples = np.concatenate((np.concatenate((xSamplesJoint, ySamplesJoint), axis=1),
                                  np.concatenate((xSamplesMarginal, ySamplesMarginal), axis=1)), axis=0)
        samples = Variable(torch.from_numpy(samples).type(torch.FloatTensor), requires_grad=False)
        lossHistory = []
        mi = 0.0
        for epoch in range(numEpochs):
            outputDiscriminator = self(samples)
            scoreJoint = outputDiscriminator[:numSamplesJoint, :]
            scoreMarginal = outputDiscriminator[numSamplesJoint:, :]
            if self.divergenceMeasure == 'JS':
                Ep = (np.log(2.0) - F.softplus(-scoreJoint)).mean()
                En = (F.softplus(-scoreMarginal) + scoreMarginal - np.log(2.0)).mean()
            elif self.divergenceMeasure == 'KL':
                Ep = scoreJoint.mean()
                En = logSumExp(scoreMarginal, 0) - np.log(scoreMarginal.size(0))
            else:
                raise NotImplementedError
            mi = Ep - En
            loss = -mi
            self.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer.step()
            lossHistory.append(loss.data.numpy())
        return np.asscalar(mi.data.numpy()), -np.array(lossHistory)
