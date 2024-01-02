from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from mine.utils import logSumExp, RegressionDataset


class MINE(nn.Module):
    def __init__(self, inputSpaceDim, archSpecs, divergenceMeasure='KL', learningRate=1e-4):
        """Initializes the MINE model.

        Args:
          inputSpaceDim: The dimension of the input space. 
          archSpecs: Architecture specifications containing layer sizes and activation functions.
          divergenceMeasure: The divergence measure to use, either 'KL' or 'JS'.
          learningRate: The learning rate for the Adam optimizer.

        The constructor initializes the neural network layers, activation functions, 
        divergence measure, and Adam optimizer according to the provided specifications.
        """
        super().__init__()
        layerSizes = archSpecs['layerSizes'] + [1]
        self.activationFunctions = archSpecs['activationFunctions']
        self.activationFunctions.append('linear')
        assert len(self.activationFunctions) == (len(layerSizes))

        self.inputHeadX = nn.Linear(inputSpaceDim, layerSizes[0])
        self.inputHeadY = nn.Linear(inputSpaceDim, layerSizes[0])

        self.layers = nn.ModuleList()
        for l in range(len(layerSizes) - 1):
            self.layers.append(nn.Linear(layerSizes[l], layerSizes[l + 1]))
        self.divergenceMeasure = divergenceMeasure
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learningRate)

    def forward(self, x, y):
        x = torch.relu(self.inputHeadX(torch.unsqueeze(x, 1)) +
                       self.inputHeadY(torch.unsqueeze(y, 1)))
        for l in range(len(self.layers)):
            activationFunction = self.activationFunctions[l + 1]
            x = self.layers[l](x) if activationFunction.lower() == 'linear' \
                else eval('torch.' + activationFunction.lower())(self.layers[l](x))
        return x

    def calcMI(self, xSamplesJoint, ySamplesJoint, xSamplesMarginal, ySamplesMarginal, numEpochs=500,
               batchSize=None, smoothCoeff=0.01):
        '''
            xSamplesJoint: Samples from the joint distribution p(x,y)
            ySamplesJoint: Samples from the joint distribution p(x,y)
            xSamplesMarginal: Samples from the marginal distribution p(x)
            ySamplesMarginal: Samples from the marginal distribution p(y)
            The mutual information is then calculated as:MI(X,Y) = Ep(x,y)[f(x,y)] - Ep(x)p(y)[f(x,y)]Where f(x,y) is the output of the trained network.
        '''
        numSamplesJoint = xSamplesJoint.shape[0]
        # assert return true or false
        assert ySamplesJoint.shape[0] == numSamplesJoint
        assert ySamplesMarginal.shape[0] == xSamplesMarginal.shape[0]

        if batchSize is None:
            batchSize = xSamplesJoint.shape[0]

        samplesJoint = Variable(torch.from_numpy(np.concatenate((xSamplesJoint, ySamplesJoint), axis=1))
                                .type(torch.FloatTensor), requires_grad=False)
        samplesMarginal = Variable(torch.from_numpy(np.concatenate((xSamplesMarginal, ySamplesMarginal), axis=1))
                                   .type(torch.FloatTensor), requires_grad=False)

        trainData = RegressionDataset(samplesJoint, samplesMarginal)
        trainLoader = DataLoader(
            dataset=trainData, batch_size=int(batchSize), shuffle=True)

        miHistory = []
        movingAverage = 1.0
        mi = 0.0
        for epoch in range(numEpochs):
            for batchJoint, batchMarginal in trainLoader:
                scoreJoint = self(
                    batchJoint[:, :batchJoint.shape[1] // 2], batchJoint[:, batchJoint.shape[1] // 2:])
                scoreMarginal = self(batchMarginal[:, :batchMarginal.shape[1] // 2],
                                     batchMarginal[:, batchMarginal.shape[1] // 2:])
                if self.divergenceMeasure == 'JS':
                    Ep = (np.log(2.0) - F.softplus(-scoreJoint)).mean()
                    En = (F.softplus(-scoreMarginal) +
                          scoreMarginal - np.log(2.0)).mean()
                elif self.divergenceMeasure == 'KL':
                    Ep = scoreJoint.mean()
                    En = logSumExp(scoreMarginal, 0) - \
                        np.log(scoreMarginal.size(0))
                else:
                    raise NotImplementedError
                mi = Ep - En
                if self.divergenceMeasure == 'KL':
                    movingAverage = (1 - smoothCoeff) * movingAverage + smoothCoeff * torch.mean(
                        torch.exp(scoreMarginal))
                    loss = -(torch.mean(scoreJoint) - (1 / movingAverage.mean()).detach() * torch.mean(
                        torch.exp(scoreMarginal)))
                else:
                    loss = -mi
                self.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.optimizer.step()
            miHistory.append(mi.item())
        return mi.item(), np.array(miHistory)
