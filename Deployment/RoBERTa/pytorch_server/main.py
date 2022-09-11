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
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW


from transformers import RobertaModel as Model
from transformers import RobertaTokenizer as Tokenizer
from transformers import get_linear_schedule_with_warmup


from http.server import HTTPServer, BaseHTTPRequestHandler
import cgi

import gc
gc.collect()
torch.cuda.empty_cache()
torch.set_num_threads(1)


# Hyperparameters
pre_trained_model = 'roberta-base'
BATCH_SIZE = 1
MAX_TOKEN_LENGTH = 100


# Set seed to ensure reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# Set device to CUDA if available
device =  'cpu' if not torch.cuda.is_available() else 'cuda'
print('Device:', device)
print(torch.version.cuda) 


# ## Load data

# Load train, test and dev data

classes = pd.read_csv('./data/AG_NEWS_DATASET/classes.csv', sep=',',header=None)


# Split sets into attributes and labels
classes = classes.to_numpy()
classes = {classes[i][0]:classes[i][1] for i in range(len(classes))}


# ## Data pre-processing

# Define a tokenizer function
tokenizer = Tokenizer.from_pretrained(pre_trained_model, do_lower_case=False)


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


class NewsClassifierDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        x = self.X[i]
        y = self.y[i]
            
        encoding = get_encoding(x, MAX_TOKEN_LENGTH, truncation=True)
        
        return {
            'X': x, 
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

print("Data loaded")


class NewsClassifierModel(nn.Module):

    def __init__(self, n_classes):
        super(NewsClassifierModel, self).__init__()
        print("Loading pretrained model")
        self.bert = Model.from_pretrained(pre_trained_model)
        print("Pretrained model loaded")
        self.drop = nn.Dropout(p=0.40)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.frezzed_bert = False
        
    def forward(self, input_ids, attention_mask):

        o = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        output = self.drop(o.get('pooler_output'))
        output = self.out(output)
        output = F.softmax(output, dim=1)
        
        return output


model = NewsClassifierModel(len(classes))

# If model checkpoint is available, load it
if os.path.exists('./checkpoint.bin'):
    model.load_state_dict(torch.load('./checkpoint.bin', map_location=device))
    print('Model loaded')
else:
    print("Model not found")

for param in model.parameters():
    param.grad = None
    
for param in model.bert.parameters():
    param.grad = None

data_loader = DataLoader(
        NewsClassifierDataset(np.array(["example"]), np.zeros(1)),
        batch_size=BATCH_SIZE,
        num_workers=0
)
model = model.to(device)
model = model.eval()
model.bert = model.bert.eval()

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
        attention_mask = sample['attention_mask'].to(device)
        output = model(input_ids = input_ids, attention_mask = attention_mask)
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
