import numpy as np
from typing import Any, Callable, List, Type
import numpy.typing as npt


class Mutation:

    def __init__(self, population: Type) -> None:
        self.popClass : Type = population
        self.execFunc : Callable = getattr(self, self.popClass.mutationFuncName)
        self.arrayFunc : Callable = getattr(self, self.popClass.mutationArrayFuncName)

    def mutate(self) -> None:
        if not self.popClass.name in self.popClass.globalVars.data["allMutatedChromossomes"]: 
            return
        mutatedChromossomes : npt.NDArray = self.popClass.globalVars.data["allMutatedChromossomes"][self.popClass.name]
        if self.popClass.arrayType.startswith("masked"):
            np.ma.apply_along_axis(self.arrayFunc,1,self.popClass.pop[mutatedChromossomes])
        else:
            np.apply_along_axis(self.arrayFunc,1,self.popClass.pop[mutatedChromossomes])
        

    def nonmasked_0(self,chromossome : npt.NDArray) -> npt.NDArray:
        mutationQtty : int = self.popClass.mutationParams["mutationQtty"]
        choosenIndexes : npt.NDArray = np.random.choice(len(chromossome),mutationQtty)
        chromossome[choosenIndexes] = self.execFunc(chromossome,mutationQtty,choosenIndexes)
        return chromossome

    def masked_0(self,chromossome : npt.NDArray) -> npt.NDArray:
        mutationQtty : int = self.popClass.mutationParams["mutationQtty"]
        nonMaskedIndexes : npt.NDArray = np.where(chromossome.mask == False)[0]
        choosenIndexes : npt.NDArray = np.random.choice(nonMaskedIndexes,mutationQtty)
        chromossome[choosenIndexes] = self.execFunc(chromossome,mutationQtty,choosenIndexes)
        return chromossome

    def percentage_MoreOrLessTax_Spectrum(self,chromossome : npt.NDArray, mutationQtty : int, choosenIndexes : npt.NDArray) -> npt.NDArray:
        mutationTax : float = self.popClass.mutationParams["mutationTax"]
        percentageArray : npt.NDArray = np.random.rand(mutationQtty) * 2 * mutationTax
        return chromossome[choosenIndexes] + percentageArray - mutationTax

    def choice_AnyGenPopValue(self,chromossome : npt.NDArray, mutationQtty : int, choosenIndexes : npt.NDArray) -> npt.NDArray:
        return np.random.choice(self.popClass.genPopParams["availableValues"],mutationQtty)
