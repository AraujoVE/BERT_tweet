import numpy as np
from numpy.random import default_rng
from typing import Any, Callable, List, Type
import numpy.typing as npt


class PreUpdatePop:

    def __init__(self,eaAlg : Type):
        self.ea : Type = eaAlg
        self.execFunc : Callable = getattr(self, self.ea.preUpdatePopFuncName)

    def preUpdatePop(self) -> None:
        self.execFunc()


    def replaceRandomPop(self) -> None:
        rng = default_rng()

        toKeepChromossomes : npt.NDArray = rng.choice(np.arange(self.ea.popSize - 1)+1,size=( self.ea.popSize - self.ea.offspringSize - 1),replace=False)
        self.ea.globalVars.setAttr("toKeepChromossomes",np.concatenate((np.array([0]),toKeepChromossomes)))

        self.ea.globalVars.setAttr("toAddOffsprings",np.arange(self.ea.offspringSize))

        standardList : npt.NDArray = np.arange(self.ea.popSize)
        np.random.shuffle(standardList[1:])
        self.ea.globalVars.setAttr("newOrder",standardList)

        print("toKeepChromossomes",end="\n\t")
        print(self.ea.globalVars.data["toKeepChromossomes"])
        print("toAddOffsprings",end="\n\t")
        print(self.ea.globalVars.data["toAddOffsprings"])
        print("newOrder",end="\n\t")
        print(self.ea.globalVars.data["newOrder"])