from BertDataclass import BertData_Initial, BertData_Hyperparameters, BertData_Fixed, BertData_Variable
from BertModel import BertModel
import json
from numpy import ma

def bertExecution(fitnessIndex,fixedArguments,trainableParams):
    #Getting fixed arguments from dict param 'fixedArguments'
    validation = fixedArguments["validation"]
    modelPath = fixedArguments["modelPath"]
    tokenizerPath = fixedArguments["tokenizerPath"]
    inputDataPath = fixedArguments["inputDataPath"]
    extraVocabPath = fixedArguments["extraVocabPath"]
    preTrainedModelPath = fixedArguments["preTrainedModelPath"]
    outputDataPath = fixedArguments["outputDataPath"]

    testMode = False #Test mode definition

    #Getting variable params from trainableparams

    batchSize = int(trainableParams[-1])
    learningRates = [float(el)*(10**-5) for el in trainableParams[:-1] if not el is ma.masked]
    epochs = len(learningRates) - 1

    bertModelParams : BertData_Initial = BertData_Initial(
        testMode,
        validation,
        modelPath,
        tokenizerPath,
        inputDataPath,
        extraVocabPath,
        preTrainedModelPath,
        outputDataPath
    )


    bertHyperparams : BertData_Hyperparameters = BertData_Hyperparameters(learningRates,batchSize,epochs)

    bertModel : BertModel = BertModel(bertModelParams)
    bertModel.setHyperparameters(bertHyperparams)
    jsonText = {
        "batchSize": batchSize,
        "epochsNo": epochs,
        "learningRates": learningRates,
        "epochs": {}
    }
    with open("statistics.json","w") as f: json.dump(jsonText,f)


    print(f"cur Values = {trainableParams}")
    bertModelResults = bertModel.train()
    with open("statistics.json","r") as f: d = f.read().strip()
    with open("statisticsFull.json","r") as fil: f = fil.read().strip()
    with open("statisticsFull.json","w") as fil: fil.write(f+"\n"+d)



    return {"index":fitnessIndex,"value":bertModelResults}
fixedParams = json.load(open("evoAlgParam.json"))["fitnessFunctionFixedArguments"]
print(fixedParams)
treinableParamsList = [
    [3, 2.6, 2.2, 1.8, 1.4, 1, 32],
    [3, 2.6, 2.2, 1.8, 1.4, 1, 64],
    [5, 4.4, 3.8, 3.2, 2.6, 2, 32],
    [5, 4.4, 3.8, 3.2, 2.6, 2, 64]
]

bestParam = {'value':None,'fit':-1.1}
for trainableParams in treinableParamsList:
    for i in range(3):
        fitnessResultCur = bertExecution(0,fixedParams,trainableParams)['value']

print(f"Best Overall = {bestParam}")
'''
'''
