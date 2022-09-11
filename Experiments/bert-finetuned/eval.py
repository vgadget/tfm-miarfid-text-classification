#!/usr/bin/env python
# coding: utf-8

# # News classification

# ## Libraries

# In[ ]:

import sys

#sys.stdout = open("output.txt", "w")
#sys.stderr = open("errors.txt", "w")

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import numpy as np
import pandas as pd

#MANAGEMENT PURPOSES ONLY-
from tqdm import tqdm
tqdm_disabled = False

# In[ ]:

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW


from transformers import BertModel as Model
from transformers import BertTokenizer as Tokenizer
from transformers import get_linear_schedule_with_warmup

pre_trained_model = 'bert-base-cased'
finetuning = False

# Hyperparameters
# In[ ]:
BATCH_SIZE =32
EPOCHS = 100
LEARNING_RATE = 2e-5
MAX_TOKEN_LENGTH = 100



import gc
gc.collect()

torch.cuda.empty_cache()


# Set seed to ensure reproducibility

# In[ ]:


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# Set device to CUDA if available

# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:')
print(device)
print(torch.version.cuda) 


# ## Load data

# Load train, test and dev data

# In[ ]:


train = pd.read_csv('./data/AG_NEWS_DATASET/train.csv', sep=',',header=None)
test = pd.read_csv('./data/AG_NEWS_DATASET/test.csv', sep=',',header=None)
dev = pd.read_csv('./data/AG_NEWS_DATASET/development.csv', sep=',',header=None)
classes = pd.read_csv('./data/AG_NEWS_DATASET/classes.csv', sep=',',header=None)


# Split sets into attributes and labels

# In[ ]:


X_train = train[[1,2]].to_numpy()
y_train = train[[0]].to_numpy().transpose().flatten()
X_test = test[[1,2]].to_numpy()
y_test = test[[0]].to_numpy().transpose().flatten()
X_dev = dev[[1,2]].to_numpy()
y_dev = dev[[0]].to_numpy().transpose().flatten()
classes = classes.to_numpy()
classes = {classes[i][0]:classes[i][1] for i in range(len(classes))}


# ## Data pre-processing

# Define a tokenizer function

# In[ ]:


tokenizer = Tokenizer.from_pretrained(pre_trained_model, do_lower_case=False)


# In[ ]:


def get_encoding(text, max_length, truncation=True):
    return tokenizer.encode_plus(
            text, 
            max_length=max_length, 
            add_special_tokens=True,
            return_token_type_ids=False, 
            padding='max_length', 
            truncation=truncation,
            return_attention_mask=True, 
            return_tensors='pt')


# ## Model


# ### Dataset class

# Let's define a dataset class in order to use it in our pytorch model

# In[ ]:


class NewsClassifierDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        x = self.X[i][0]+' '+self.X[i][1]
        y = self.y[i]
            
        encoding = get_encoding(x, MAX_TOKEN_LENGTH, truncation=True)
        
        return {
            'X': x, 
            'y': torch.tensor(y, dtype=torch.long),
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


# In[ ]:


num_workers = 0

train_data_loader = DataLoader(
    NewsClassifierDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    num_workers=num_workers
)

development_data_loader = DataLoader(
    NewsClassifierDataset(X_dev, y_dev),
    batch_size=BATCH_SIZE,
    num_workers=num_workers
)

test_data_loader = DataLoader(
    NewsClassifierDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
    num_workers=num_workers
)

print("Data loaded")

# ### Model class

# In[ ]:


class NewsClassifierModel(nn.Module):

    def __init__(self, n_classes):
        super(NewsClassifierModel, self).__init__()
        self.bert = Model.from_pretrained(pre_trained_model)
        self.drop = nn.Dropout(p=0.40)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.frezzed_bert = False
        
    def forward(self, input_ids, attention_mask, freeze_bert=True):
        
        
        # Turn on/off the BERT layers
        # Useefull for finetuning

        # Freeze BERT layers if freeze_bert is True and not previously frozen
        if freeze_bert and not self.frezzed_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            self.frezzed_bert = True
        
        # Unfreeze BERT layers if freeze_bert is False and previously frozen
        elif not freeze_bert and self.frezzed_bert:
            for param in self.bert.parameters():
                param.requires_grad = True
            self.frezzed_bert = False
        

        o = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        
        output = self.drop(o.get('pooler_output'))
        output = self.out(output)
        output = F.softmax(output, dim=1)
        
        return output


# In[ ]:


model = NewsClassifierModel(len(classes))
model = model.to(device)

# If model checkpoint is available, load it
if os.path.exists('./checkpoint.bin'):
    model.load_state_dict(torch.load('./checkpoint.bin'))
    print('Model loaded')

print("Model created")

# ## Training

# In[ ]:


optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = CrossEntropyLoss().to(device)


# In[ ]:

def criterion(output, target):

    y_true = torch.zeros_like(output)
    
    for i in range(len(target)):
        y_true[i][target[i].item()-1] = 1

    return loss_fn(output, y_true)


# In[ ]:


scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=0,
    num_training_steps = (len(train_data_loader) * EPOCHS)
)


# In[ ]:

def eval(model, data_loader, confusion_mat=False):

    n_correct = 0
    n_samples = 0
    loss = 0
    M = {}

    model = model.eval()

    with torch.no_grad():
        for c in classes.keys():
            M[c] = {}
            for i in classes.keys():
                M[c][i] = 0
        for sample in data_loader:
            input_ids = sample['input_ids'].to(device)
            attention_mask = sample['attention_mask'].to(device)
            targets = sample['y'].to(device)

            output = model(input_ids = input_ids, attention_mask = attention_mask)

            _, predicted = torch.max(output, dim=1)
            predicted += torch.ones_like(predicted)

            for i in range(len(predicted)):
                t = targets[i].item()
                p = predicted[i].item()

                M[t][p] += 1


            loss = criterion(output, targets)
            n_correct += torch.sum(predicted == targets)
            n_samples += len(targets)

        accuracy = 100 * (n_correct.double() / n_samples)
        avg_recall = 0
        avg_precis = 0

        output_string = "\n"

        for c in classes.keys():
            output_string += "\t"+ str(c)

        output_string += "\t"+ "Recall"

        for c in classes.keys():
            output_string += "\n"+"-"*50 + "\n"
            output_string += str(c) + "|"
            recall = M[c][c]/sum([M[c][i] for i in classes.keys()])
            avg_recall += recall
            for i in classes.keys():
                output_string += "\t"+ str(M[c][i])

            output_string += "\t"+"{:.2f}".format(recall*100)

        output_string += "\n"+"-"*50 +"\nPrec."
        for c in classes.keys():
            precision = M[c][c]/sum([M[i][c] for i in classes.keys()])
            avg_precis += precision
            output_string += "\t"+"{:.2f}".format(precision*100)

        output_string += "\n\n Avg. accuracy \t Avg. Prec. \t Avg. Recall \t Avg. F1-Sc\n"
        avg_recall = (avg_recall/len(classes.keys()))*100
        avg_precis = (avg_precis/len(classes.keys()))*100
        f1 = (2*avg_recall*avg_precis)/(avg_recall+avg_precis)
        output_string += "{:.2f}".format(accuracy) + "\t" + "{:.2f}".format(avg_precis) + "\t" + "{:.2f}".format(avg_recall) + "\t" + "{:.2f}".format(f1)
        output_string += "\n\n TP \t TN \t FN \t FP\n" + "-"*40

        for c in classes.keys():
            TP = M[c][c]
            FN = sum([M[c][i] for i in classes.keys() if c != i])
            FP = sum([M[i][c] for i in classes.keys() if c != i])
            TN = sum([M[i][j] for i in classes.keys() for j in classes.keys() if i != c and j != c])
            output_string += f"\n {TP} \t {TN} \t {FN} \t {FP}"


        if confusion_mat:
           return accuracy, loss, output_string, n_samples

    return accuracy, loss


# In[ ]:


def epoch_step(model, freeze_bert=False):

    epoch_loss = []
    n_correct = 0
    n_samples = 0

    model = model.train()

    for sample in tqdm(train_data_loader, desc='Training...', disable=tqdm_disabled):
       
        optimizer.zero_grad()
        
        input_ids = sample['input_ids'].to(device)
        attention_mask = sample['attention_mask'].to(device)
        targets = sample['y'].to(device)

        output = model(input_ids = input_ids, attention_mask = attention_mask, freeze_bert=freeze_bert)
        
        _, predicted = torch.max(output, dim=1)
        predicted += torch.ones_like(predicted)

        loss = criterion(output, targets)

        epoch_loss.append(loss.item())

        n_correct += torch.sum(predicted == targets)
        n_samples += len(targets)

        loss.backward()

        """
        Gradient clipping
        clip_grad_norm() performs gradient clipping. 
        It is used to mitigate the problem of exploding gradients, 
        which is of particular concern for recurrent networks (which LSTMs are a type of).
        """
        clip_grad_norm(model.parameters(), max_norm = 1.0)

        optimizer.step()
        scheduler.step()


    accuracy = 100 * (n_correct.double() / n_samples)
    return accuracy, np.mean(epoch_loss)


# In[ ]:


train_history = []
dev_history = []


# In[ ]:

# ### Test 

# Evaluate model on test set

# In[ ]:


test_acuracy, test_loss, M, L = eval(model, development_data_loader, confusion_mat=True)
print(f"Loss: {test_loss:.4f},  Accuracy: {test_acuracy:.4f}%")
print(M)
print(L)

