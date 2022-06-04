from typing import TYPE_CHECKING, Any, Callable, List, Type
import numpy.typing as npt
import numpy as np
import random

if TYPE_CHECKING:
    from BERT.EvolutionaryAlgorithm.population import Population

class Crossover:

    #Initialization
    def __init__(self, population: 'Population') -> None:
        self.popClass : 'Population' = population #Getting parent class
        self.execFunc : Callable = getattr(self, self.popClass.crossoverFuncName) #Specific crossover operation (meanValue,randomValue,etc)
        self.arrayFunc : Callable = getattr(self, self.popClass.crossoverArrayFuncName) #General crossover func (masked,nonmasked,etc)

    #Base crossover call
    def crossover(self) -> None:
        self.arrayFunc()

    #Mean of each parental value
    def meanValue(self,parentals : npt.NDArray) -> npt.NDArray:
        return parentals.mean(axis = 0)    

    #Random value for each gene from one parental - just nonmasked
    def randomValue(self,parentals : npt.NDArray) -> npt.NDArray:
        parentalsPerGene : npt.NDArray = np.random.randint(low=0,high=np.shape(parentals)[0],size=np.shape(parentals)[1]) #Get a vector of random values between '0' and 'no. of parentals'. Vector size equals to 'features size'
        return np.choose(parentalsPerGene,parentals) #choose from parentals the random list created


    #Masked general function
    def masked_0(self) -> None:
        parentals : npt.NDArray = self.popClass.pop[self.popClass.globalVars.data["parentalsIndex"]] #Parentals that will generate offspring
        parentalsMaskedValuesList : npt.NDArray = np.count_nonzero(parentals.mask == True, axis = 1) #For each parental, get the number of masked values
        
        parentalsMaskAvg : float = np.average(parentalsMaskedValuesList) #Get avg masked value size
        masksNo : int = round(parentalsMaskAvg) - random.getrandbits(1) if parentalsMaskAvg % 1 == 0.5 else round(parentalsMaskAvg) #Decide if 'x.5' avg values will became 'x' or 'x+1'
        
        offspring : npt.NDArray = self.execFunc(parentals) #Execute specific function
        if masksNo != 0: offspring.mask[-masksNo:] = True #Mask the calculated masksNo size
        
        self.popClass.offsprings.append(offspring) #Append the offspring

    def nonmasked_0(self) -> None:
        parentals : npt.NDArray = self.popClass.pop[self.popClass.globalVars.data["parentalsIndex"]] #Parentals that will generate offspring
        self.popClass.offsprings.append(self.execFunc(parentals)) #Append the calculated offspring
