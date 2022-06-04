import numpy as np
from numpy.random import default_rng
from typing import Any, Callable, List, Type
import numpy.typing as npt


class StopCondition:
    def __init__(self,eaAlg : Type):
        self.ea : Type = eaAlg
        self.execFunc : Callable = getattr(self, self.ea.stopConditionFuncName)

    def stop(self) -> bool:
        return self.execFunc()

    def basicCount(self):
        return self.ea.i >= self.ea.stopConditionParams["maxIter"]
    
    def valueGreater(self):
        return False if not ("bestIndividual" in self.ea.globalVars.data) else self.ea.globalVars.data["bestIndividual"]["value"] > self.ea.stopConditionParams["threshold"]