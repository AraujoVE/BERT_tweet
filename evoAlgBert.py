from EvolutionaryAlgorithm.evoAlgIter import EvoAlgIter
from BertExecution import bertExecution

from random import random
import json

from numpy import ma

def mockFitnessFunc(fixedParams,variableParams): return variableParams[0] / (sum([el for el in variableParams[:-1] if el != -1]) + 1)

evoAlgIter = EvoAlgIter(mockFitnessFunc,"./evoAlgParam.json") 
#evoAlgIter = EvoAlgIter(bertExecution,"./evoAlgParam.json") 
bestParams = evoAlgIter.run()

batchSize = bestParams["params"][-1]
learningRates = [el for el in bestParams["params"][:-1] if not el is ma.masked]
epochs = len(learningRates) - 1
bestParamsObj = {
    "batchSize" : batchSize,
    "learningRates" : learningRates,
    "epochs" : epochs,
    "prediction" : bestParams["value"]
}

with open('bestParams.json', 'w') as outfile:
    json.dump(bestParamsObj, outfile)