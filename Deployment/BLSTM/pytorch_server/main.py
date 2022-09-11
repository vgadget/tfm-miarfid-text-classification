#!/usr/bin/env python
# coding: utf-8


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TRANSFORMERS_CACHE'] = './cache/'

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer



from http.server import HTTPServer, BaseHTTPRequestHandler
import cgi

import gc
gc.collect()
torch.cuda.empty_cache()
torch.set_num_threads(1)


# Hyperparameters
BATCH_SIZE = 1
MAX_TOKEN_LENGTH = 100
HIDDEN_DIM = 256
NUM_LAYERS = 6
MAX_WORDS = 6000



# Set seed to ensure reproducibility
SEED = 32
np.random.seed(SEED)
torch.manual_seed(SEED)


# Set device to CUDA if available
device =  'cpu' if not torch.cuda.is_available() else 'cuda'
print('Device:', device)
print(torch.version.cuda) 


# ## Load data
train = pd.read_csv('./data/AG_NEWS_DATASET/train.csv', sep=',',header=None)
test = pd.read_csv('./data/AG_NEWS_DATASET/test.csv', sep=',',header=None)
dev = pd.read_csv('./data/AG_NEWS_DATASET/development.csv', sep=',',header=None)
classes = pd.read_csv('./data/AG_NEWS_DATASET/classes.csv', sep=',',header=None)

test = test[20:40]

# Load train, test and dev data
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
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts([x[0]+' '+x[1] for x in X_train])

def get_encoding(text, truncation=True):
    sequences = tokenizer.texts_to_sequences([text])
    return sequence.pad_sequences(sequences, maxlen=MAX_TOKEN_LENGTH, padding='post') if not truncation else sequences



class NewsClassifierDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        x = self.X[i]
        y = self.y[i]
            
        encoding = get_encoding(x, truncation=False)   

        return {
            'X': x, 
            'y': torch.tensor(round(y), dtype=torch.long),
            'input_ids': encoding.flatten()        
            }


class NewsClassifierModel(nn.Module):

    def __init__(self, n_classes):
        super(NewsClassifierModel, self).__init__()
        self.dropout = nn.Dropout(p=0.40)
        self.embedding = nn.Embedding(MAX_WORDS, HIDDEN_DIM, padding_idx=0)
        self.lstm = nn.LSTM(input_size=HIDDEN_DIM, hidden_size=HIDDEN_DIM, num_layers=NUM_LAYERS, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(in_features=HIDDEN_DIM*2, out_features=HIDDEN_DIM) # Bidirectional
        self.out = nn.Linear(in_features=HIDDEN_DIM, out_features=n_classes)
        
    def forward(self, x):

        # Hidden and cell state definion
        h = torch.zeros((NUM_LAYERS*2, x.shape[0], HIDDEN_DIM)).to(device) # Bidirectional
        c = torch.zeros((NUM_LAYERS*2, x.shape[0], HIDDEN_DIM)).to(device) # Bidirectional

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

model = NewsClassifierModel(len(classes))

# If model checkpoint is available, load it
if os.path.exists('./checkpoint.bin'):
    model.load_state_dict(torch.load('./checkpoint.bin', map_location=device))
    print('Model loaded')
else:
    print("Model not found")

for param in model.parameters():
    param.grad = None

data_loader = DataLoader(
        NewsClassifierDataset(np.array(["example"]), np.zeros(1)),
        batch_size=BATCH_SIZE,
        num_workers=0
)
model = model.to(device)
model = model.eval()

print("Model created")


def predict(model, texts=[]):
    predictions = []
    
    data_loader = DataLoader(
        NewsClassifierDataset(np.array(texts), np.zeros(len(texts))),
        batch_size=BATCH_SIZE,
        num_workers=0
    )
    
    for sample in data_loader:
            
        input_ids = sample['input_ids'].to(device)
        output = model(input_ids)
        
        _, predicted = torch.max(output, dim=1)
        predicted += torch.ones_like(predicted)
        
        for i in range(len(predicted)):
            predictions.append(predicted[i].item())

    return predictions


print("Model ready")

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
        
    def do_POST(self):
        ctype, pdict = cgi.parse_header(self.headers.get('content-type'))
        pdict['boundary'] = bytes(pdict['boundary'], 'utf-8')
        
        fields = cgi.parse_multipart(self.rfile, pdict)
        text = fields.get("text")[0]

        p = predict(model, [f"{text}"])[0]
        c = classes[p]        
        
        html = f"{c}"
        
        self.send_response(200)
        self.end_headers()
        self.wfile.write(bytes(html, "utf-8"))


print('Server listening')

httpd = HTTPServer(('0.0.0.0', 8080), SimpleHTTPRequestHandler)
httpd.serve_forever()


print("Server started")
