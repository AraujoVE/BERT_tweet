import numpy as np
from numpy.random import default_rng
from typing import Any, Callable, List, Type
import numpy.typing as npt


class Selection:

    def __init__(self,eaAlg : Type):
        self.ea : Type = eaAlg
        self.execFunc : Callable = getattr(self, self.ea.selectionFuncName)

    def select(self) -> None:
        self.ea.globalVars.setAttr("parentalsIndex",[])
        self.execFunc()

    def tournment(self) -> None:

        rng = default_rng()

        for i in range(self.ea.selectionParams["parentalsNo"]):
            candidates : npt.NDArray = rng.choice(self.ea.popSize, size=self.ea.selectionParams["popPerTournment"], replace=False) #Getting random indexes from a fitnessArray
            print(f"candidates {i} : {candidates}")
            bestIndex : npt.NDArray = np.argmax(self.ea.globalVars.data["fitnessArray"][candidates]) #Getting 'candidates' index that has the best fitness
            self.ea.globalVars.data["parentalsIndex"].append(candidates[bestIndex]) #Getting best fitness index
        self.ea.globalVars.data["parentalsIndex"] = np.array(self.ea.globalVars.data["parentalsIndex"])
        print("parIndex :",end="\n\t")
        print(self.ea.globalVars.data["parentalsIndex"])
