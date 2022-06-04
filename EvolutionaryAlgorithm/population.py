from EvolutionaryAlgorithm.crossover import Crossover
from EvolutionaryAlgorithm.mutation import Mutation
from EvolutionaryAlgorithm.globalVars import GlobalVars

from EvolutionaryAlgorithm.genPop import GenPop

from typing import Callable, List, Type, Dict, Any

import numpy as np

import numpy.typing as npt

class Population:

    def __init__(self, globalVars : Type[GlobalVars], popName : str, orderId : int, specificPopParams : Dict[str,Any], genericPopParams : Dict[str,Any]) -> None:

        self.orderId = orderId
        self.globalVars : Type[GlobalVars] = globalVars

        self.popSize : int = genericPopParams["size"]
        self.genericParams : Dict[str,Any] = genericPopParams["params"]

        self.name : str = popName

        self.arrayType : Callable = specificPopParams["popType"]
        self.genPopParams : str = specificPopParams["genPop"]["params"]
        self.genPopFuncName : str = specificPopParams["genPop"]["funcName"]
        self.genPopArrayFuncName : str = specificPopParams["genPop"]["arrayFuncName"]
        self.genPopFunc : Type[GenPop] = GenPop(self)

        self.crossoverParams : str = specificPopParams["crossover"]["params"]
        self.crossoverFuncName : str = specificPopParams["crossover"]["funcName"]
        self.crossoverArrayFuncName : str = specificPopParams["crossover"]["arrayFuncName"]
        self.crossoverFunc : Type[Crossover] = Crossover(self)

        self.mutationParams : str = specificPopParams["mutation"]["params"]
        self.mutationFuncName : str = specificPopParams["mutation"]["funcName"]
        self.mutationArrayFuncName : str = specificPopParams["mutation"]["arrayFuncName"]
        self.mutationFunc : Type[Mutation] = Mutation(self)

        self.offsprings : npt.NDArray = []


    def initializePop(self) -> None:
        self.pop = self.genPopFunc.genPop()

    def order(self) -> None:
        self.pop = self.pop[self.globalVars.data["orderArray"]]

    
    def crossover(self) -> None:
        self.crossoverFunc.crossover()

    def mutate(self) -> None:
        self.mutationFunc.mutate()
    
    def updatePop(self) -> None :
        self.offsprings = np.ma.masked_array(self.offsprings) if self.arrayType.startswith("masked") else np.array(self.offsprings) 
        self.pop = self.pop[self.globalVars.data["toKeepChromossomes"]]
        toAddOffsprings : npt.NDArray = self.offsprings[self.globalVars.data["toAddOffsprings"]] 
        self.pop = np.ma.vstack((self.pop,toAddOffsprings)) if self.arrayType.startswith("masked") else np.vstack((self.pop,toAddOffsprings))
        self.pop = self.pop[self.globalVars.data["newOrder"]]
        self.offsprings = []
























