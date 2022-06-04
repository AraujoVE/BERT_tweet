from dataclasses import dataclass
import typing
from numpy import typing as npt


from typing import List, Any

@dataclass
class BertData_Initial:
    """
    Class to hold the fixed data from the Bert model
    """
    testMode : bool = False

    validation : float = 0.2

    modelPath : str = "bert-model-path"
    tokenizerPath : str = "bert-model-path"


    inputDataPath : str = ""
    extraVocabPath : str = ""
    preTrainedModelPath : str = ""
    outputDataPath : str = ""

@dataclass
class BertData_Hyperparameters:
    """
    Class to hold the Bert hyperparameters data from the Bert model
    """
    learningRates : List[float] = None
    batchSize : int = 32
    epochs : int = 1

@dataclass
class BertData_Fixed:
    """
    Class to hold the Bert variable data from the Bert model
    """
    tokenizer : Any = None
    model : Any = None
    device : Any = None

@dataclass
class BertData_Variable:
    """
    Class to hold the Bert variable data from the Bert model
    """
    inputValues : npt.NDArray = None
    expandedInputLabels : npt.NDArray = None
    inputLabels : npt.NDArray = None
    attentionMasks : npt.NDArray = None

    trainDataloader : Any = None
    validationDataloader : Any  = None

    optimizer : Any = None

    learningRates : npt.NDArray = None