from collections import defaultdict
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from os.path import exists

# for data visualisation and preprocessing
import seaborn as sn
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# for nlp preprocessing
import nltk
from nltk.tokenize import word_tokenize

# for nlp model
import torch
import torchvision
from torchvision.transforms import transforms
from torchvision import models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

# for nlp training
import torch.optim as optim
from torch.optim import Adam

# for data transformations
from transformers import T5ForConditionalGeneration, AutoTokenizer, Adafactor

torch.manual_seed(42)
SEED = 1

class LSTM(nn.Module):
    def __init__(self, num_vocab, num_class, dropout=0.3) :
        super().__init__()

        self.embedding = nn.Embedding(num_vocab, 128, padding_idx=0)

        # LSTM
        self.lstm = nn.LSTM(128, 64, batch_first=True)

        self.ff1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.ff2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(32, num_class)
        )

    def forward(self, x, hidden):
        """
        The forward method takes in the input and the previous hidden state 
        """
        h0 = self.embedding(x)
        h1, hidden = self.lstm(h0, hidden)
        h2 = self.ff1(h1)
        h3 = self.ff2(h2) # get output of last token in seq
        p = F.log_softmax(h3[:, -1], dim=1)
        return p, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, 64), torch.zeros(1, batch_size, 64))

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path)["model_state_dict"])

def train(
        model, 
        dataset, 
        batch_size, 
        learning_rate, 
        num_epoch, 
        device, 
        model_path=None,
        loss_path=None,
        restart=False
    ):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    criterion = nn.NLLLoss() # equivalent to cross entropy for log softmax
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    losses = []
    n = 0
    if not restart:
        if model_path != None and exists(model_path):
            load_model(model, model_path)
        if loss_path != None and exists(loss_path):
            losses = torch.load(loss_path)
            if len(losses) > 0:
                n = losses[-1][0]
                num_epoch = num_epoch

    h0, c0 =  model.init_hidden(batch_size)

    start = datetime.datetime.now()

    for epoch in range(n, num_epoch):
        model.train()
        running_loss = 0.0
        for step, data in enumerate(data_loader, 0):
            # get the inputs; data is a tuple of (inputs, labels)
            texts = data[0].to(device)
            labels = data[1].to(device)
            
            # zero the parameter gradients
            model.zero_grad()

            # do forward propagation
            log_probs, hidden = model.forward(texts, (h0, c0))

            # do loss calculation
            loss = criterion(log_probs, labels)

            # do backward propagation
            loss.backward()
            
            # do parameter optimization step
            optimizer.step()

            # calculate running loss value for non padding
            running_loss += loss.item()

            # print loss value every 100 steps and reset the running loss
            if step % 10 == 9:
                losses.append((epoch + 1, step + 1, running_loss / 10))
                print('[%d, %5d] loss: %.3f' %
                    losses[-1])
                running_loss = 0.0

            # define the checkpoint and save it to the model path
        checkpoint = {
            "text_vocab": dataset.text_vocab,
            "label_vocab": dataset.label_vocab,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": criterion,
        }
        torch.save(checkpoint, model_path)
        print('Model saved in ', model_path)
        torch.save(losses, loss_path)
        print("Losses saved in ", loss_path)

    end = datetime.datetime.now()

    print('Training finished in {} minutes.'.format((end - start).seconds / 60.0))

# prediction function, returns list
def predict(model, model_path, dataset, batch_size, device, classes=None, output_path=None):
    load_model(model, model_path)
    model.eval()

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    output_ls = []
    h0, c0 =  model.init_hidden(batch_size)
    
    with torch.no_grad():
        for texts, labels, original_texts in data_loader:
            texts = texts.to(device)
            outputs, hiddens = model(texts, (h0, c0))
            outputs = outputs.cpu()
            # the label with the highest energy will be our prediction
            for i in range(len(texts)):
                original_text_i = original_texts[i]
                label_i = labels[i].item()
                predicted_i = torch.argmax(outputs[i].squeeze()).item()
                if classes is not None:
                    label_i = classes[label_i]
                    predicted_i = classes[predicted_i]
                output_ls.append((original_text_i, label_i, predicted_i))

    output_df = pd.DataFrame.from_records(output_ls, columns =['Text', 'Target', 'Predicted'])

    if output_path is not None:
        output_df.to_csv(output_path, index=False)

    return output_df