#Importing Libraries
import torch

import numpy as np
import pandas as pd

import os
import json


from transformers import AutoModel, AutoTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup, BertTokenizer, RobertaForSequenceClassification, RobertaTokenizer

import random
import time

from typing import Any, List
from numpy import typing as npt


from DataLoaders import DataLoaders
from TestTrain import TestTrain

from BertDataclass import BertData_Initial, BertData_Hyperparameters, BertData_Fixed, BertData_Variable


class BertModel():
    bertDataInitial : BertData_Initial = BertData_Initial()
    bertDataHyperparameters : BertData_Hyperparameters = BertData_Hyperparameters()
    bertDataFixed : BertData_Fixed = BertData_Fixed()
    bertDataVariable : BertData_Variable = BertData_Variable()
    dataLoader : DataLoaders
    trainingIteration : TestTrain

    def __init__(self,bertData : BertData_Initial):
        self.bertDataInitial = bertData
        self.setFixedDataRoberta() #Generate tokenize, model and device
        self.incrementVocab() #Increase vocabulary
        self.tokenizeSentences() #Tokenize the sentences


    #Generate the model, tokenizer and device
    def setFixedDataBert(self) -> None:
        tokenizer : Any = BertTokenizer.from_pretrained(self.bertDataInitial.tokenizerPath,do_lower_case=False)
        model = BertForSequenceClassification.from_pretrained(
            self.bertDataInitial.modelPath, #Use the chosen model.
            num_labels = 2, #The number of output labels--2 for binary classification. You can increase this for multi-class tasks.   
            output_attentions = False, #Whether the model returns attentions weights.
            output_hidden_states = False, #Whether the model returns all hidden-states.
        )   
        device : Any = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") #Trying to use cuda, if possilbe, else use cpu
        if torch.cuda.is_available(): 
            model.cuda()     
            print("\n\n\nCUDA ON")
        else:
            print("\n\n\nCUDA OFF")

        self.bertDataFixed = BertData_Fixed(tokenizer,model,device) #Create the fixed data class
        return

    #Generate the model, tokenizer and device
    def setFixedDataRoberta(self) -> None:
        tokenizer : Any = RobertaTokenizer.from_pretrained(self.bertDataInitial.tokenizerPath,do_lower_case=False)
        model = RobertaForSequenceClassification.from_pretrained(
            self.bertDataInitial.modelPath, #Use the chosen model.
            num_labels = 2, #The number of output labels--2 for binary classification. You can increase this for multi-class tasks.   
            output_attentions = False, #Whether the model returns attentions weights.
            output_hidden_states = False #Whether the model returns all hidden-states.
        )   
        device : Any = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") #Trying to use cuda, if possilbe, else use cpu
        if torch.cuda.is_available(): 
            model.cuda()     
            print("\n\n\nCUDA ON")
        else:
            print("\n\n\nCUDA OFF")

        self.bertDataFixed = BertData_Fixed(tokenizer,model,device) #Create the fixed data class
        return





    #Increase vocabulary
    def incrementVocab(self) -> None:
        if self.bertDataInitial.testMode or self.bertDataInitial.extraVocabPath == '': return #Return when not in test mode or when extra vocab not present
        newTokens = list(pd.read_csv(self.bertDataInitial.extraVocabPath)["vocab"]) #Get the new tokens
        if len(newTokens) > 0:
            self.bertDataFixed.tokenizer.add_tokens(newTokens) #Add the new tokens to the tokenizer
            self.bertDataFixed.model.resize_token_embeddings(len(self.bertDataFixed.tokenizer)) #Resize the embeddings to fit the new vocabulary

    #Tokenize the sentence and set the input labels when not in test mode
    def tokenizeSentences(self) -> None:
        baseData : pd.DataFrame = pd.read_csv(self.bertDataInitial.inputDataPath,delimiter="|") #Read the input data
        sentences : List[str] = list(baseData["sentence"]) #Get the sentences
        inputValues : List[npt.NDArray] = [] #Create the input values
        attentionMasks : List[npt.NDArray] = [] #Create the attention masks
        maxTokenSize : int = min(int(1.25*(max([len(self.bertDataFixed.tokenizer.tokenize(sentence)) for sentence in sentences]) + 2)),512)
        for sentence in sentences:
            encodedDict =  self.bertDataFixed.tokenizer.encode_plus( #Encode the sentence
                sentence,                     # Sentence to encode.
                add_special_tokens = True,    # Add '[CLS]' and '[SEP]'
                max_length = maxTokenSize,             # Pad & truncate all sentences.
                padding='max_length',
                pad_to_max_length = True,     # Do padding
                return_attention_mask = True, # Construct attn. masks.
                return_tensors = 'pt',        # Return pytorch tensors.
            )
            inputValues.append(np.array(encodedDict['input_ids'])) #Get the input token IDs.
            attentionMasks.append(np.array(encodedDict['attention_mask'])) #Get the attention mask.

        #Convert to numpy arrays
        self.bertDataVariable.inputValues = np.array(inputValues)
        self.bertDataVariable.attentionMasks = np.array(attentionMasks)

        if not self.bertDataInitial.testMode: 
          self.bertDataVariable.inputLabels = np.array(baseData["appreciative"]) #Get the labels if not in test mode
          self.bertDataVariable.expandedInputLabels = np.array(baseData["expandedLabel"]) #Get the expanded labels (BolsonaroAppreciative(0),BolsonaroNonAppreciative(1),LulaAppreciative(2),LulaNonAppreciative(3),BothAppreciative(4),BothNonAppreciative(5)) if not in test mode




    ###############################################################################################

    #Function called to set the hyperparameters
    def setHyperparameters(self,hyperParams : BertData_Hyperparameters) -> None:
        self.bertDataHyperparameters : BertData_Hyperparameters = hyperParams
        return

    #Generate the learning rates 
    def learningRateGen(self) -> None:
        leaningRatesEpoch : List[float] = self.bertDataHyperparameters.learningRates #Basic learning rates
        learningRates : npt.NDArray = np.linspace(leaningRatesEpoch[:-1],leaningRatesEpoch[1:],len(self.bertDataVariable.trainDataloader)+1)[:-1].transpose().flatten() #Filling learning rates in between
        self.bertDataVariable.learningRates = learningRates #Set the learning rates


    #Save the model
    def savePreTrained(self):
        #Create output directory if needed
        if not os.path.exists(self.bertDataInitial.preTrainedModelPath):
            os.makedirs(self.bertDataInitial.preTrainedModelPath)

        print("Saving model to %s" % self.bertDataInitial.preTrainedModelPath)

        model_to_save = self.bertDataFixed.model.module if hasattr(self.bertDataFixed.model, 'module') else self.bertDataFixed.model  #Take care of distributed/parallel training
        model_to_save.save_pretrained(self.bertDataInitial.preTrainedModelPath) #Save the model
        self.bertDataFixed.tokenizer.save_pretrained(self.bertDataInitial.preTrainedModelPath) #Save the tokenizer

    def updateData(self,predictions : npt.NDArray) -> None:
        oldData = pd.read_csv(self.bertDataInitial.inputDataPath) #Get the new tokens
        oldData["prediction"] = predictions #Add the new tokens to the tokenizer
        oldData.to_csv(self.bertDataInitial.outputDataPath,index=False) #Save the new data
        return





    def train(self):
        #return if in test mode
        if self.bertDataInitial.testMode: 
            print("\nModel in Test Mode, no training will be done\n")
            return

        self.dataLoader = DataLoaders(self.bertDataInitial,self.bertDataHyperparameters, self.bertDataVariable,self.bertDataFixed) #Create the data loader
        self.trainingIteration = TestTrain(self.bertDataInitial,self.bertDataHyperparameters, self.bertDataVariable,self.bertDataFixed) #Create the training iteration

        self.bertDataVariable.trainDataloader, self.bertDataVariable.validationDataloader = self.dataLoader.train() #Get train and validation dataloaders
        self.learningRateGen() #Generate the learning rates

        self.bertDataVariable.optimizer = AdamW(self.bertDataFixed.model.parameters(),lr = self.bertDataHyperparameters.learningRates[0], eps = 1e-8 ) #Set the optimizer and its parameters
        

        #Needed?
        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        #Needed?

        fitness = self.trainingIteration.train() #Train the model and get the fitness

        #Save the model if there is no validation
        if not bool(self.bertDataInitial.validation): 
            self.savePreTrained()
            return None

        return fitness #Else, the model is being trained and must return the fitness

    def test(self):
        #return if in train mode
        if not self.bertDataInitial.testMode: 
            print("\nModel in Test Mode, no training will be done\n")
            return

        self.bertDataFixed.model.to(self.bertDataFixed.device) #Move the model to the device

        self.dataLoader = DataLoaders(self.bertDataInitial,self.bertDataHyperparameters, self.bertDataVariable,self.bertDataFixed) #Create the data loader
        self.trainingIteration = TestTrain(self.bertDataInitial,self.bertDataHyperparameters, self.bertDataVariable,self.bertDataFixed) #Create the training iteration

        self.bertDataVariable.testDataloader = self.dataLoader.test() #Get the test dataloader
        self.bertDataFixed.model.eval() #Set the model to evaluation mode
        predictions : npt.NDArray = self.trainingIteration.test() #Test the model
        self.updateData(predictions) #Update the data