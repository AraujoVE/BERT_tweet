import numpy as np
from numpy.random import default_rng
from typing import Any, Callable, List, Type, Dict
import numpy.typing as npt


class PreMutation:

    def __init__(self,eaAlg : Type):
        self.ea : Type = eaAlg
        self.execFunc : Callable = getattr(self, self.ea.preMutationFuncName)

    def preMutate(self) -> None:
        self.execFunc()

    def basic(self) -> None:
        popNames : List[str] = []
        popMutOccurances : List[float] = []
        for pop in self.ea.pops:
            popNames.append(pop.name)
            popMutOccurances.append(pop.mutationParams["mutationOccurance"])

        rng = default_rng()


        chosenMutations : npt.NDArray = rng.choice(popNames,self.ea.preMutationParams["chromossomesMutatedPerPop"],p=popMutOccurances)

        chosenIndexes : npt.NDArray = rng.choice(np.array(self.ea.popSize - 1) + 1,self.ea.preMutationParams["chromossomesMutatedPerPop"], replace=False)

        allMutatedChromossomes : Dict[str,npt.NDArray]= {}

        for mut in set(chosenMutations): allMutatedChromossomes[mut] = [] 
        for i in range(len(chosenIndexes)): allMutatedChromossomes[chosenMutations[i]].append(chosenIndexes[i])
        for mut in set(chosenMutations): allMutatedChromossomes[mut] = np.array(allMutatedChromossomes[mut])
        self.ea.globalVars.setAttr("allMutatedChromossomes",allMutatedChromossomes)
        print("allMutatedChromossomes",end="\n\t")
        print(allMutatedChromossomes)
