from typing import Any, Callable, List, Type
import numpy.typing as npt
import numpy as np


class GenPop:

    def __init__(self, population: Type) -> None:
        self.popClass : Type = population
        self.execFunc : Callable = getattr(self, self.popClass.genPopFuncName)
        self.arrayFunc : Callable = getattr(self, self.popClass.genPopArrayFuncName)

    def genPop(self) -> npt.NDArray:
        return self.arrayFunc()

    #####################################################################################
    def minMaxValues(self,popSize,genesSize) -> npt.NDArray:
        return np.random.randint(low = int(self.popClass.genPopParams["minGeneValue"]), high = int(self.popClass.genPopParams["maxGeneValue"])+1, size = (popSize, genesSize)) #Creating random pop with attributes of len of vecMaxLength

    def inListValues(self,popSize,genesSize) -> npt.NDArray:
        return np.random.choice(self.popClass.genPopParams["availableValues"], size = (popSize, genesSize))
    #####################################################################################

    #####################################################################################
    def maskTerms_0(self,joinedArray : npt.NDArray) -> npt.NDArray:
        #How many masked values +1 due to the last column being the own mask value number 
        maskQtty : int = int(joinedArray[-1]) + 1 
        
        #Setting the masked values to '-1'
        joinedArray[-maskQtty:] = -1.0

        #Returning all but the last column
        return joinedArray[:-1]

    def masked_0(self) -> npt.NDArray:
        popSize : int = self.popClass.popSize
        maxGenesNo : int = self.popClass.genPopParams["maxGenesNo"]
        minGenesNo : int = self.popClass.genPopParams["minGenesNo"]

        maxMaskedQttyInArray : int = maxGenesNo - minGenesNo #Setting max number of masked values.

        fullPop : npt.NDArray = self.execFunc(popSize,maxGenesNo)
        masksQttyArray : npt.NDArray = np.random.randint(low = 0, high = maxMaskedQttyInArray+1, size = (popSize, 1)) #Creating random pop with attributes of len of vecMaxLength 
        joinedArray : npt.NDArray = np.append(fullPop,masksQttyArray,1) #Joining both matrixes to further calculations
        
        maskedFullPop : npt.NDArray = np.apply_along_axis(self.maskTerms_0, 1, joinedArray) #Masking values in respect to each row with '-1'
        maskedFullPop = maskedFullPop.astype(float) #Converting to float
        maskedFullPop = np.ma.masked_array(maskedFullPop) #Converting to masked array
        maskedFullPop = np.ma.masked_where(maskedFullPop == -1.0, maskedFullPop) #Masking '-1' values
        return maskedFullPop
    #####################################################################################

    #####################################################################################
    def nonmasked_0(self) -> npt.NDArray:
        return self.execFunc(self.popClass.popSize,self.popClass.genPopParams["genesSize"])
    #####################################################################################

