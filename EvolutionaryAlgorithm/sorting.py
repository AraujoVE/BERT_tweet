import numpy as np
from typing import Any, Callable, List, Type
import numpy.typing as npt

class Sorting:

    def __init__(self,eaAlg : Type):
        self.ea : Type = eaAlg
        self.execFunc : Callable = getattr(self, self.ea.sortingFuncName)

    def sort(self) -> None:
        self.ea.globalVars.setAttr("orderArray",np.arange(self.ea.popSize))
        self.execFunc()

    def bestFirst(self) -> None:
        bestFitIndex : int = np.argmax(self.ea.globalVars.data["fitnessArray"]) #Getting index of most fit chromossome

        #If best elem is not on pos 0, swap first elem with best fitness elem
        if bestFitIndex != 0:
            self.ea.globalVars.data["orderArray"][[0,bestFitIndex]] = self.ea.globalVars.data["orderArray"][[bestFitIndex,0]]
            self.ea.globalVars.data["fitnessArray"][[0,bestFitIndex]] = self.ea.globalVars.data["fitnessArray"][[bestFitIndex,0]]
            for pop in self.ea.pops: pop.order()

