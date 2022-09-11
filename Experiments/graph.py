#!/usr/bin/env python
# coding: utf-8

# In[12]:


from cProfile import label
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


# In[13]:


## read parameters
_ , MODEL_NAME = sys.argv

MODEL_NAME_FINETUNED = MODEL_NAME + '-finetuned'

# In[14]:


# Read files from the current directory
dev_history_base = []
dev_history_finetuned = []
train_history_base = []
train_history_finetuned = []


with open(f"{MODEL_NAME}/dev_history.txt", "r") as f:
    current_line = 0
    for line in f:
        if current_line > 0:
            epoch, accuracy, loss = line.split(",")
            dev_history_base.append(float(accuracy))
        current_line += 1

with open(f"{MODEL_NAME}/train_history.txt", "r") as f:
    current_line = 0
    for line in f:
        if current_line > 0:
            epoch, accuracy, loss = line.split(",")
            train_history_base.append(float(accuracy))
        current_line += 1

if "lstm" not in MODEL_NAME:
    
    with open(f"{MODEL_NAME_FINETUNED}/train_history.txt", "r") as f:
        current_line = 0
        for line in f:
            if current_line > 0:
                epoch, accuracy, loss = line.split(",")
                train_history_finetuned.append(float(accuracy))
            current_line += 1
    with open(f"{MODEL_NAME_FINETUNED}/dev_history.txt", "r") as f:
        current_line = 0
        for line in f:
            if current_line > 0:
                epoch, accuracy, loss = line.split(",")
                dev_history_finetuned.append(float(accuracy))
            current_line += 1
        


# In[15]:


len(dev_history_base)


# In[16]:


len(train_history_base)


# In[17]:

training_color = 'orangered'
dev_color = 'royalblue'


plt.plot(train_history_base, color=training_color)
plt.plot(dev_history_base, color=dev_color)

if "lstm" not in MODEL_NAME:
    plt.plot(train_history_finetuned, color=training_color, linestyle='-.')
    plt.plot(dev_history_finetuned, color=dev_color, linestyle='-.')

plt.title(f"Accuracy evolution for {MODEL_NAME} model")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(labels=["training frozen base model", "development frozen base model", "training fine-tuned model", "development fine-tuned model"], loc='lower right')
plt.show()
