from BertDataclass import BertData_Initial, BertData_Hyperparameters, BertData_Fixed, BertData_Variable
from typing import Callable, List, Any, Tuple
from numpy import typing as npt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torchsample.samplers import StratifiedSampler 
import pandas as pd


class DataLoaders():

    def __init__(self,bertDataInitial : BertData_Initial, bertDataHyperparameters : BertData_Hyperparameters, bertDataVariable : BertData_Variable, bertDataFixed : BertData_Fixed) -> None:
        self.bertDataInitial = bertDataInitial
        self.bertDataHyperparameters = bertDataHyperparameters
        self.bertDataVariable = bertDataVariable
        self.bertDataFixed = bertDataFixed
        return

    #Split the data into train and validation in a stratified way
    def stratifiedSplit(self) -> Tuple:
        #Getting the parameters to local vars
        inputValues : npt.NDArray = self.bertDataVariable.inputValues
        labels : npt.NDArray = self.bertDataVariable.inputLabels
        expandedLabels : npt.NDArray = self.bertDataVariable.expandedInputLabels
        attentionMasks : npt.NDArray = self.bertDataVariable.attentionMasks

        sss : Any = StratifiedShuffleSplit(n_splits=1, test_size=self.bertDataInitial.validation, random_state=0) #Set stratified split
        sss.get_n_splits(inputValues, expandedLabels) #Make the split with inputValues in respect to labels classes

        setTensors : Callable = lambda list,index: torch.cat([torch.tensor(value) for value in list[index]], dim = 0)

        for train_index, test_index in sss.split(inputValues, expandedLabels):
            #Setting values for train and validation
            trainValues : Any = setTensors(inputValues,train_index)
            validationValues : Any = setTensors(inputValues, test_index)

            #Setting attention masks for train and validation
            trainAttentionMasks : Any = setTensors(attentionMasks, train_index)
            validationAttentionMasks : Any = setTensors(attentionMasks, test_index)
            
            #Setting labels for train and validation
            trainExpandedLabels : Any = torch.tensor(expandedLabels[train_index])
            trainLabels : Any = torch.tensor(labels[train_index])
            validationLabels : Any = torch.tensor(labels[test_index])
            break

        #Defining the dataloaders
        return trainValues, trainAttentionMasks, trainLabels, trainExpandedLabels, validationValues, validationAttentionMasks, validationLabels


    #Setting train Dataloader
    def trainDataloader(self,trainDataset,trainExpandedLabels,batchSize) -> DataLoader:
        trainDataloader : DataLoader = DataLoader(
            trainDataset, #Training samples
            sampler = StratifiedSampler(class_vector=trainExpandedLabels,batch_size=batchSize), #Select batches stratified randomly in respect to the expanded sampling labels
            batch_size = batchSize #Trains with this batch size.
        )
        return trainDataloader

    #Setting validation Dataloader
    def valDataloader(self,validationDataset,batchSize) -> DataLoader:
        validationDataloader : DataLoader = DataLoader(
            validationDataset, #Validation samples.
            sampler = SequentialSampler(validationDataset), #Pull out batches sequentially.
            batch_size = batchSize #Evaluate with this batch size.
        )
        return validationDataloader


    #Generate and return the dataloaders
    def train(self) -> Tuple[DataLoader,DataLoader]:
        batchSize : int = self.bertDataHyperparameters.batchSize #Batch size
        
        trainValues, trainAttentionMasks, trainLabels, trainExpandedLabels, validationValues, validationAttentionMasks, validationLabels = self.stratifiedSplit() #Split the data into train and validation in a stratified way
        
        trainDataset : Any = TensorDataset(trainValues, trainAttentionMasks, trainLabels) #Create the train tensor dataset
        validationDataset : Any = TensorDataset(validationValues, validationAttentionMasks, validationLabels) #Create the validation tensor dataset
        
        trainingDataloader : DataLoader = self.trainDataloader(trainDataset,trainExpandedLabels,batchSize) #Setting train Dataloader
        validationDataloader : DataLoader = self.valDataloader(validationDataset, batchSize) #Setting validation Dataloader
        
        return trainingDataloader, validationDataloader #Return the dataloaders