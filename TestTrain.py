from BertDataclass import BertData_Initial, BertData_Hyperparameters, BertData_Fixed, BertData_Variable
import numpy as np
import datetime
import time
import torch
from sklearn.metrics import matthews_corrcoef
from numpy import typing as npt
from typing import List, Any

class TestTrain:
    def __init__(self,bertDataInitial : BertData_Initial, bertDataHyperparameters : BertData_Hyperparameters, bertDataVariable : BertData_Variable, bertDataFixed : BertData_Fixed) -> None:
        self.bertDataInitial : BertData_Initial = bertDataInitial
        self.bertDataHyperparameters : BertData_Hyperparameters = bertDataHyperparameters
        self.bertDataVariable : BertData_Variable = bertDataVariable
        self.bertDataFixed : BertData_Fixed = bertDataFixed
        self.generalIter : int = 0
        self.totalTrainLoss : int = 0
        return

    #Function to calculate the accuracy of our predictions vs labels
    def flatAccuracy(self,preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)


    #Takes a time in seconds and returns a string hh:mm:ss
    def formatTime(self,elapsed):
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    #For reps in reps print log
    def progressLog(self,step,t0):
        if step % 40 == 0 and not step == 0:
            elapsed = self.formatTime(time.time() - t0)        
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len( self.bertDataVariable.trainDataloader), elapsed))

    def printTrainTime(self,avg_train_loss,t0):
        trainingTime = self.formatTime(time.time() - t0)
        print("\n  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(trainingTime))
        return trainingTime

    #Print traing data
    def printTraining(self,epoch_i):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.bertDataHyperparameters.epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        return time.time()



    #Function to make a batch iteration training
    def trainBatchIter(self,step,batch,t0) -> None:
        self.generalIter += 1
        self.progressLog(step,t0)

        #This line bellow is not working. device problems, some tensors in cuda and others in cpu 
        inputIds, inputMask, labels = tuple(t.to(self.bertDataFixed.device) for t in batch) # Unpack the batch (id,mask and labels)
        #print(inputIds)
        #print(inputMask)
        #print(labels)

        self.bertDataFixed.model.zero_grad() #Clears any previously calculated gradients before performing a backward pass     

        (loss,logits) = self.bertDataFixed.model(
            inputIds, 
            token_type_ids=None, 
            attention_mask=inputMask, 
            labels=labels,
            return_dict=False
        ) #Perform a forward pass

        self.totalTrainLoss += loss.item() # Accumulate the training loss over all of the batches to calculate the average loss at the end.

        loss.backward() #Perform a backward pass to calculate the gradients.

        torch.nn.utils.clip_grad_norm_(self.bertDataFixed.model.parameters(), 1.0) #Clip the norm of the gradients to 1.0. to prevent "exploding gradients"

        self.bertDataVariable.optimizer.step() #Update parameters and take a step using the computed gradient. The optimizer dictates the "update rule"

        #Update the learning rate
        if self.generalIter != len(self.bertDataVariable.learningRates):
          for g in self.bertDataVariable.optimizer.param_groups: g['lr'] = self.bertDataVariable.learningRates[self.generalIter]


    #Function to make a batch iteration validation
    def valBatchIter(self,batch):#,device,model,total_eval_loss,predictions,trueValues):#,totalEvalAccuracy):
        inputIds, inputMask, labels = tuple(t.to(self.bertDataFixed.device) for t in batch) # Unpack the batch (id,mask and labels)

        
        with torch.no_grad(): #Don't compute graph during the forward pass, since this is only needed for backprop        
            (loss, logits) = self.bertDataFixed.model(
                inputIds, 
                token_type_ids=None, 
                attention_mask=inputMask,
                labels=labels,
                return_dict=False
            ) #Perform a forward pass
            
        #Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        self.predictions.append(np.array(logits))
        self.trueLabels.append(np.array(label_ids))

        #totalEvalAccuracy += flatAccuracy(logits, label_ids) #Calculate the accuracy for this batch of test sentences, and accumulate it over all batches.

    def testBatchIter(self,batch):
        inputIds, inputMask = tuple(t.to(self.bertDataFixed.device) for t in batch) # Unpack the batch (id and mask)
    
        with torch.no_grad():# Forward pass, calculate logit predictions
            outputs = self.bertDataFixed.model(
                inputIds, 
                token_type_ids=None, 
                attention_mask=inputMask
            )

        logits = outputs[0]

        #Move logits to CPU
        logits = logits.detach().cpu().numpy()
    
        return logits


    #Get the MCC final score
    def fullMatthewsCoef(self):
        #Concatenate the predictions
        flat_predictions = np.concatenate(self.predictions, axis=0) #Combine the results across all batches. 

        #Get the max value of the list
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten() #For each sample, pick the label (0 or 1) with the higher score.

        #Concatenate de true labels
        flat_true_labels = np.concatenate(self.trueLabels, axis=0) #Combine the correct labels for each batch into a single list.

        #Return the matthews coef
        return matthews_corrcoef(flat_true_labels, flat_predictions) #Calculate the MCC




    def validate(self,epoch_i,t0):
        avg_train_loss = self.totalTrainLoss / len(self.bertDataVariable.trainDataloader) #Calculate the average loss over all of the batches.
        
        training_time = self.printTrainTime(avg_train_loss,t0) #Measure how long this epoch took.
            
        print("\nRunning Validation...")

        t0 = time.time()

        self.bertDataFixed.model.eval() #Put the model in evaluation mode

        self.totalEvalLoss = 0

        #Evaluate data for one epoch
        self.predictions, self.trueLabels = [],[]

        #Calculate eval loss and append prediction and true labels from each batch to future calculations
        for batch in self.bertDataVariable.validationDataloader:
            self.valBatchIter(batch)

        self.totalEvalAccuracy = self.fullMatthewsCoef() 





    def train(self):
        self.trainingStats = []
        total_t0 = time.time()

        applyValidation : bool = bool(self.bertDataInitial.validation)
        self.generalIter = 0
        ii = 0
        for epoch_i in range(self.bertDataHyperparameters.epochs):
            #print("\t\t#",ii)
            ii+= 1

            t0 = self.printTraining(epoch_i)
            self.totalTrainLoss = 0 #Reset the total loss for this epoch.

            self.bertDataFixed.model.train() #Put the model into training mode.

            #For each batch of training data...
            for step, batch in enumerate(self.bertDataVariable.trainDataloader):
                self.trainBatchIter(step,batch,t0)

            if applyValidation: self.validate(epoch_i,t0)

        return self.totalEvalAccuracy

    def test(self) -> npt.NDArray:
        predictions : List[npt.NDArray] = []
        for batch in self.bertDataVariable.testDataloader:
            logits = self.testBatchIter(batch)
            predictions.append(np.arry(logits))
        
        return np.concatenate(predictions,axis=0)