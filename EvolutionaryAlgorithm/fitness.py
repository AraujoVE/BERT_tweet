import numpy as np
from numpy.random import default_rng
from typing import Any, Callable, List, Type, Dict
import numpy.typing as npt


class Fitness:

    #Initialization
    def __init__(self,eaAlg : Type):
        self.ea : Type = eaAlg
        self.execFunc : Callable = getattr(self, self.ea.fitnessFuncName)

    #Base fitness call
    def fit(self) -> None:
        self.fullPop : npt.NDArray = np.ma.hstack(tuple([Pop.pop for Pop in self.ea.pops])) #Setting populations together
        print(f"\n\nfpop1 =\n{self.fullPop}")
        self.fullPop = np.split(self.fullPop,len(self.fullPop)) #Splitting each individual
        self.ea.globalVars.setAttr("bestIndividual",self.execFunc()) #Setting the bestIndividual

    #All values are calculated except the first - which is the best individual
    def dontRecalcBest(self) -> Dict[str,Any]:
        fitnessArray : List = []
        startingIndex : int = 1 if "fitnessArray" in self.ea.globalVars.data else 0 #In the first iteration, the individual 0 is calculated, in the others, not
        for individual in self.fullPop[startingIndex:] : fitnessArray.append(self.ea.calcFitnessFunc(self.ea.fixedArguments,list(individual[0]))) #Calculating each individual fitness by given fitness function
        if "fitnessArray" in self.ea.globalVars.data: fitnessArray.insert(0,self.ea.globalVars.data["fitnessArray"][0]) #In the second and consecutive iterations, the older best fitness is just added - not calculated
        fitnessArray = np.array(fitnessArray) #np conversion
        self.ea.globalVars.setAttr("fitnessArray",fitnessArray) #Loading into globalVars

        #bestIndividual dict values
        bestIndividual : Dict[str,Any] = {
            "params" : list(self.fullPop[np.argmax(fitnessArray)][0]),
            "value" : np.amax(fitnessArray)
        }
        for i in range(len(self.fullPop)):
            print(f"Individual:\n{self.fullPop[i]}\n\tValue:{fitnessArray[i]}")
        print(f"bestIndividual = {bestIndividual}")
        return bestIndividual
