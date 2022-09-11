from tkinter import Y
import matplotlib.pyplot as plt
import os

# Get name of all folders in current directory
folders = [f.path for f in os.scandir() if f.is_dir() and f.name != '.' and f.name != '..' and not  f.name.startswith('.')]

#sort list by length of list
folders.sort(key=len)

def read_history(folder):
    train_history = []
    with open(f"{folder}/train_history.txt", "r") as f:
        current_line = 0
        for line in f:
            if current_line > 0:
                epoch, accuracy, loss = line.split(",")
                train_history.append((float(accuracy), float(loss)))
            current_line += 1
    
    dev_history = []
    with open(f"{folder}/dev_history.txt", "r") as f:
        current_line = 0
        for line in f:
            if current_line > 0:
                epoch, accuracy, loss = line.split(",")
                dev_history.append((float(accuracy), float(loss)))
            current_line += 1
    
    return train_history, dev_history


# Plot all graphs
plt.figure(figsize=(20, 10))
for i, folder in enumerate(folders):
    model_name = folder[2:]
    if "finetuned" not in folder and "lstm" not in folder:
        train_history, dev_history = read_history(folder)
        if max([x[0] for x in dev_history]) > 30:
            plt.plot(range(len(dev_history)), [x[0] for x in dev_history], label=model_name)

plt.title("Accuracy on development set", y=1.03, fontsize=18)
plt.suptitle("Frozen pre-trained models", fontsize=10)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()