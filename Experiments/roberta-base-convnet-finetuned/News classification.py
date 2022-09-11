#!/usr/bin/env python
# coding: utf-8

# # News classification

# ## Libraries

# In[ ]:

import sys

sys.stdout = open("output.txt", "w")
sys.stderr = open("errors.txt", "w")

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


from transformers import RobertaModel as Model
from transformers import RobertaTokenizer as Tokenizer
from transformers import get_linear_schedule_with_warmup

pre_trained_model = 'roberta-base'
finetuning = True

# Hyperparameters
# In[ ]:
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 2e-8
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
        self.img_size = 16
        self.bert = Model.from_pretrained(pre_trained_model)
        self.drop = nn.Dropout(p=0.40)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, self.img_size**2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3, 3), stride=1, padding="same") # (16, 16)->(16, 16)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=(3, 3), stride=1, padding=0) #  (16, 16)->(14, 14)
        self.conv2_drop = nn.Dropout2d(p=0.40)
        self.conv3 = nn.Conv2d(in_channels=9, out_channels=16, kernel_size=(3, 3), stride=1, padding="same") # (14, 14)->(14, 14)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1, padding=0) # (14, 14)-->(12, 12)
        self.conv4_drop = nn.Dropout2d(p=0.40)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2) # (12, 12)->(6, 6)
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding="same") # (6, 6)->(6, 6)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding=0) # (6, 6)->(3, 3)
        self.conv6_drop = nn.Dropout2d(p=0.40)
        ## Flattern but keep the batch dimension intact
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1) 
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc4_drop = nn.Dropout(p=0.40)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 64)
        self.fc6_drop = nn.Dropout(p=0.40)
        self.fc7 = nn.Linear(64, 32)
        self.out = nn.Linear(32, n_classes)
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
        output = self.fc1(output)
        #Reshape the output to be a square image
        output = output.view(output.shape[0], 1, self.img_size, self.img_size)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.conv2_drop(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv4_drop(output)
        output = self.maxpool(output)
        output = self.conv5(output)
        output = self.conv6(output)
        output = self.conv6_drop(output)
        output = self.flatten(output)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.fc4(output)
        output = self.fc4_drop(output)
        output = self.fc5(output)
        output = self.fc6(output)
        output = self.fc6_drop(output)
        output = self.fc7(output)
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


def eval(model, data_loader):
    
    n_correct = 0
    n_samples = 0
    loss = 0

    model = model.eval()

    with torch.no_grad():
        for sample in data_loader:
            input_ids = sample['input_ids'].to(device)
            attention_mask = sample['attention_mask'].to(device)
            targets = sample['y'].to(device)

            output = model(input_ids = input_ids, attention_mask = attention_mask)
            
            _, predicted = torch.max(output, dim=1)
            predicted += torch.ones_like(predicted)
            
            loss = criterion(output, targets)
            n_correct += torch.sum(predicted == targets)
            n_samples += len(targets)

        accuracy = 100 * (n_correct.double() / n_samples)

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


best_dev_accuracy = -1
for epoch in tqdm(range(EPOCHS), desc="Epochs", disable=tqdm_disabled):

    print(f"Training... {epoch+1}/{EPOCHS}")

    epoch_step(model, freeze_bert=(not finetuning))

    accuracy, loss = eval(model, train_data_loader)
    dev_acuracy, dev_loss = eval(model, development_data_loader)
    
    train_history.append((epoch+1, accuracy, loss))
    dev_history.append((epoch+1, dev_acuracy, dev_loss))

    print(f"Epoch: {epoch+1}/{EPOCHS}, Loss: {loss:.4f}, Train accuracy: {accuracy:.4f}%, Dev accuracy:{dev_acuracy:.4f}%")
    if dev_acuracy > best_dev_accuracy:
        torch.save(model.state_dict(), 'checkpoint.bin')
        best_accuracy = dev_acuracy


# ### Test 

# Evaluate model on test set

# In[ ]:


test_acuracy, test_loss = eval(model, test_data_loader)
print(f"Loss: {test_loss:.4f},  Accuracy: {test_acuracy:.4f}%")


# In[ ]:


# save the training history
with open('train_history.txt', 'w') as f:
    f.write("Epoch, Accuracy, Loss\n")
    for epoch, accuracy, loss in train_history:
        f.write(f"{epoch},{accuracy},{loss}\n")

with open('dev_history.txt', 'w') as f:
    f.write("Epoch, Accuracy, Loss\n")
    for epoch, accuracy, loss in dev_history:
        f.write(f"{epoch},{accuracy},{loss}\n")

