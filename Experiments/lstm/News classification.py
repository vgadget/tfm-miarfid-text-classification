#!/usr/bin/env python
# coding: utf-8

# # News classification

# ## Libraries

# In[1]:

import sys

sys.stdout = open("output.txt", "w")
sys.stderr = open("errors.txt", "w")


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import get_linear_schedule_with_warmup

import numpy as np
import pandas as pd

#MANAGEMENT PURPOSES ONLY-
from tqdm import tqdm
tqdm_disabled = False
import gc
gc.collect()
torch.cuda.empty_cache()


# In[ ]:


BATCH_SIZE = 32
HIDDEN_DIM = 256
NUM_LAYERS = 6
EPOCHS = 100
LEARNING_RATE = 2e-5


# Set seed to ensure reproducibility

# In[4]:


SEED = 32
np.random.seed(SEED)
torch.manual_seed(SEED)


# Set device to CUDA if available

# In[5]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.version.cuda) 


# ## Load data

# Load train, test and dev data

# In[6]:


train = pd.read_csv('./data/AG_NEWS_DATASET/train.csv', sep=',',header=None)
test = pd.read_csv('./data/AG_NEWS_DATASET/test.csv', sep=',',header=None)
dev = pd.read_csv('./data/AG_NEWS_DATASET/development.csv', sep=',',header=None)
classes = pd.read_csv('./data/AG_NEWS_DATASET/classes.csv', sep=',',header=None)


# In[7]:


train[0:3]


# In[8]:


test[0:3]


# In[9]:


dev[0:3]


# In[10]:


classes


# Split sets into attributes and labels

# In[11]:


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

# In[12]:


MAX_TOKEN_LENGTH = 100


# In[13]:


from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer


# In[14]:


"""from matplotlib import pyplot as plt
sentences_len = [len(sentence[0]+''+sentence[1]) for sentence in X_train]  
sentences_dict = {}
for i in range(len(sentences_len)):
    if sentences_len[i] not in sentences_dict:
        sentences_dict[sentences_len[i]] = 0
    sentences_dict[sentences_len[i]] += 1

plt.bar(list(sentences_dict.keys()), list(sentences_dict.values()))
"""


# In[15]:


MAX_WORDS = 6000


# In[16]:


tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts([x[0]+' '+x[1] for x in X_train])

def get_encoding(text, truncation=True):
    sequences = tokenizer.texts_to_sequences([text])
    return sequence.pad_sequences(sequences, maxlen=MAX_TOKEN_LENGTH, padding='post') if not truncation else sequences


# In[17]:


#print(get_encoding('This is a test'))


# Let's see an example of tokenization

# In[18]:


example_text = X_train[0][0]


# In[19]:


encoding = get_encoding(example_text, truncation=False)


# In[20]:


example_text


# In[21]:


#print(encoding)


# Now we need to find the apropiate token lentgth for our tokenizer

# As we can see, a good token length will be greater than 80. So let's use 100 to ensure enough context for the training.

# ## Model

# ### Dataset class

# Let's define a dataset class in order to use it in our pytorch model

# In[23]:


class NewsClassifierDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        x = self.X[i][0]+' '+self.X[i][1]
        y = self.y[i]
            
        encoding = get_encoding(x, truncation=False)   

        return {
            'X': x, 
            'y': torch.tensor(y, dtype=torch.long),
            'input_ids': encoding.flatten()        
            }


# In[24]:


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


# In[25]:


sample = next(iter(train_data_loader))
sample.keys()
#print(len(sample['X']))    
#print(sample['y'].shape)    
#print(sample['input_ids'].shape)


# ### Model class

# In[26]:


class NewsClassifierModel(nn.Module):

    def __init__(self, n_classes):
        super(NewsClassifierModel, self).__init__()
        self.dropout = nn.Dropout(p=0.40)
        self.embedding = nn.Embedding(MAX_WORDS, HIDDEN_DIM, padding_idx=0)
        self.lstm = nn.LSTM(input_size=HIDDEN_DIM, hidden_size=HIDDEN_DIM, num_layers=NUM_LAYERS, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(in_features=HIDDEN_DIM, out_features=HIDDEN_DIM)
        self.out = nn.Linear(in_features=HIDDEN_DIM, out_features=n_classes)
        
    def forward(self, x):

        # Hidden and cell state definion
        h = torch.zeros((NUM_LAYERS, x.shape[0], HIDDEN_DIM)).to(device)
        c = torch.zeros((NUM_LAYERS, x.shape[0], HIDDEN_DIM)).to(device)

        # Initialization fo hidden and cell states
        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)

        # Embedding layer
        out = self.embedding(x)

        # Feed LSTMs
        # x -> (batch_size, seq_len, embedding_dim)
        out, (hidden, cell) = self.lstm(out, (h,c))
        out = self.dropout(out)
        
        # The last hidden state is taken
        out = torch.relu_(self.fc(out[:,-1,:]))
        out = self.dropout(out)
        out = self.out(out)

        out = torch.softmax(out, dim=1, dtype=torch.float)

        return out


# In[27]:


model = NewsClassifierModel(len(classes))
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = CrossEntropyLoss().to(device)

#if previous checkpoint is found, load it
if os.path.isfile('./checkpoint.bin'):
    print('Loading checkpoint')
    checkpoint = torch.load('./checkpoint.bin')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('Loaded checkpoint')


# ## Training

# In[29]:


def criterion(output, target):

    y_true = torch.zeros_like(output)
    
    for i in range(len(target)):
        y_true[i][target[i].item()-1] = 1

    return loss_fn(output, y_true)


# In[30]:


scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=0,
    num_training_steps = (len(train_data_loader) * EPOCHS)
)


# In[31]:


def eval(model, data_loader):
    
    n_correct = 0
    n_samples = 0
    loss = 0

    model = model.eval()

    with torch.no_grad():
        for sample in data_loader:
            input_ids = sample['input_ids'].to(device)
            targets = sample['y'].to(device)

            output = model(input_ids)
            
            _, predicted = torch.max(output, dim=1)
            predicted += torch.ones_like(predicted)
            
            loss = criterion(output, targets)
            n_correct += torch.sum(predicted == targets)
            n_samples += len(targets)

        accuracy = 100 * (n_correct.double() / n_samples)

    return accuracy, loss


# In[32]:


def epoch_step(model, freeze_bert=False):

    epoch_loss = []
    n_correct = 0
    n_samples = 0

    model = model.train()

    for sample in tqdm(train_data_loader, desc='Training...', disable=tqdm_disabled):
        
        input_ids = sample['input_ids'].to(device)
        targets = sample['y'].to(device)

        output = model(input_ids)
        
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
        #clip_grad_norm(model.parameters(), max_norm = 1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


    accuracy = 100 * (n_correct.double() / n_samples)
    return accuracy, np.mean(epoch_loss)


# In[33]:


train_history = []
dev_history = []


# In[34]:


best_dev_accuracy = -1
for epoch in tqdm(range(EPOCHS), desc="Epochs", disable=tqdm_disabled):

    accuracy, loss = epoch_step(model, freeze_bert=False)
    dev_acuracy, dev_loss = eval(model, development_data_loader)
    
    train_history.append((epoch+1, accuracy, loss))
    dev_history.append((epoch+1, dev_acuracy, dev_loss))

    print(f"Epoch: {epoch+1}/{EPOCHS}, Loss: {loss:.4f}, Train accuracy: {accuracy:.2f}%, Dev accuracy:{dev_acuracy:.2f}%")

    if dev_acuracy > best_dev_accuracy:
        torch.save(model.state_dict(), 'checkpoint.bin')
        best_accuracy = dev_acuracy


# ### Test 

# Evaluate model on test set

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


#Load the best model
model.load_state_dict(torch.load('checkpoint.bin'))
test_acuracy, test_loss = eval(model, test_data_loader)
print(f"Loss: {test_loss:.4f},  Accuracy: {test_acuracy:.4f}%")

