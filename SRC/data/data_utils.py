from collections import defaultdict
import numpy as np
import pandas as pd
import re

# for nlp preprocessing
import nltk
from nltk.tokenize import word_tokenize

# for nlp model
import torch
from torch.utils.data import Dataset



torch.manual_seed(42)
SEED = 1

def resample_data(data_path, save_path=None):
    data = pd.read_csv(data_path)

    # check that the count is balanced
    positive_data = data[data["Sentiment"] == 1]
    positive_count = positive_data.shape[0]
    negative_data = data[data["Sentiment"] == -1]
    negative_count = negative_data.shape[0]

    # resample to ensure even number of each category
    if positive_count > negative_count:
        resample = np.random.randint(0, negative_data.shape[0], positive_count - negative_count)
        new_negative_data = pd.concat([negative_data, negative_data.iloc[resample]])
        new_data = pd.concat([positive_data, new_negative_data])
    elif negative_count > positive_count:
        resample = np.random.randint(0, positive_data.shape[0], negative_count - positive_count)
        new_positive_data = pd.concat([positive_data, positive_data.iloc[resample]])
        new_data = pd.concat([new_positive_data, negative_data])

    if save_path is not None:
        new_data.to_csv(save_path, index=False)

    return new_data

class TwitterDataset(Dataset): 
    def __init__(self, texts, training, labels=None, min_freq=0, vocab=None, seq_len=32):
        """
        Read texts and get vocab
        If labels != None, indicates training, vocab will be generated from text.
        If labels == None, path to vocab file must be given
        Args:
            texts (iter): Iterable collection of texts
            labels (iter): Iterable collection of labels
            min_freq (int): Minimum frequency of words in text to be part of vocab.
            vocab (string): Path to the vocabulary file
            seq_len (int): Length of input sequence.
        """
        # download tokenizer 
        nltk.download('punkt')

        self.original_texts = texts.tolist()
        self.texts = []
        # processes and splits words
        for i in range(len(texts)):
            processed_text = word_tokenize(self.clean_text(texts.iloc[i]))
            self.texts.append(processed_text)
        self.labels = labels
        if labels is not None:
            self.labels = labels.tolist()
        self.min_freq = min_freq
        self.seq_len = seq_len

        if training: # if training
            assert labels is not None, "Labels must be given for training"
            # create text vocab
            self.text_vocab = {
                "<PAD>": 0, # padding
                "<SOS>": 1, # start of sentence
                "<EOS>": 2, # end of sentence
                "<UNK>": 3  # unknown tokens
            }
            self.text_vocab = self.create_text_vocab(self.texts, self.text_vocab)
            # create label vocab
            self.label_vocab = {}
            self.create_label_vocab(self.labels, self.label_vocab)

        else: # if testing
            assert vocab is not None , "Testing must use vocab from training"
            checkpoint = torch.load(vocab)
            self.text_vocab = checkpoint["text_vocab"]
            self.label_vocab = checkpoint["label_vocab"]

    ### ------------------- Start of helper funcs ------------------ ###

    # remove links and @ mentions
    def clean_text(self, text):
        link_re_pattern = "https?:\/\/t.co/[\w]+"
        mention_re_pattern = "@\w+"
        text = re.sub(link_re_pattern, "", text)
        text = re.sub(mention_re_pattern, "", text)
        return text.lower()

    # create text vocab
    def create_text_vocab(self, texts, vocab):
        freq = defaultdict(int)
        # count frequency of each token
        for text in texts:
            for token in text:
                freq[token] += 1
        # convert to vocab
        for token in freq:
            # only add if meets min freq
            if freq[token] >= self.min_freq:
                vocab[token] = len(vocab)
        return vocab

    # create label vocab
    def create_label_vocab(self, labels, vocab):
        for label in labels:
            if label not in vocab:
                vocab[label] = len(vocab)

    ### --------------------- End of helper funcs -------------------- ###
    
    # encode text to bag of words vector
    # truncates/pads to fit seq_len
    def make_bow_vector(self, text):
        # truncates if needed
        text = text[:self.seq_len-2] # -2 for SOS and EOS
        # encodes
        x = []
        for token in text:
            if token in self.text_vocab:
                x.append(self.text_vocab[token])
            else:
                x.append(self.text_vocab["<UNK>"])
        # sos, eos
        x = [self.text_vocab["<SOS>"]] + x + [self.text_vocab["<EOS>"]]
        # padding if needed
        n_pads = self.seq_len - len(x)
        x = x + [self.text_vocab["<PAD>"]] * n_pads

        return torch.LongTensor(x)

    def vocab_size(self):
        """
        A function to inform the vocab size. The function returns two numbers:
            num_vocab: size of the vocabulary
            num_class: number of class labels
        """
        num_vocab = len(self.text_vocab)
        num_class = len(self.label_vocab)
        return num_vocab, num_class

    def __len__(self):
        """
        Return the number of instances in the data
        """
        return len(self.texts)

    def __getitem__(self, i):
        """
        Return the i-th instance in the format of:
            (text, label)
        Text and label should be encoded according to the vocab (word_id).

        DO NOT pad the tensor here, do it at the collator function.
        """
        text = self.make_bow_vector(self.texts[i])
        label = torch.empty(1)
        if self.labels != None: # if training
            label = self.label_vocab[self.labels[i]]
        
        return text, label, self.original_texts[i]