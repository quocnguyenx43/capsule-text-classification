import pandas as pd
import numpy as np

from transformers import AutoModel, AutoTokenizer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report

import warnings
warnings.filterwarnings("ignore")


print("Reading data")
train_df = pd.read_csv('./vihsd/train.csv')
dev_df = pd.read_csv('./vihsd/dev.csv')
test_df = pd.read_csv('./vihsd/test.csv')

#pre-process
import re
import numpy as np

USE_CUDA = True
STOPWORDS = './vietnamese-stopwords.txt'
with open(STOPWORDS, "r") as ins:
    stopwords = []
    for line in ins:
        dd = line.strip('\n')
        stopwords.append(dd)
    stopwords = set(stopwords)

def filter_stop_words(train_sentences, stop_words):
    new_sent = [word for word in train_sentences.split() if word not in stop_words]
    train_sentences = ' '.join(new_sent)

    return train_sentences

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'', text)

def preprocess(text, tokenizer):
    # text = filter_stop_words(text, stopwords)
    text = deEmojify(text)
    text = text.lower()
    text = tokenizer(text, return_tensors='pt', max_length=200, truncation=True, padding='max_length')
    ids = text['input_ids']
    attn_mask = text['attention_mask']
    return ids, attn_mask


def full_preprocess(X, tokenizer):
    X_p = [preprocess(text, tokenizer) for text in X]
    X_ids = [x[0] for x in X_p]
    X_attn_mask = [x[1] for x in X_p]
    X_ids, X_attn_mask = torch.cat(X_ids), torch.cat(X_attn_mask)
    return X_ids, X_attn_mask


# Define a custom Dataset class
class TextDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F

class ConvLayer(nn.Module):

    def __init__(self, in_channels=1, out_channels=256, kernel_size=9):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1)

    def forward(self, x):
        return F.relu(self.conv(x))

class PrimaryCaps(nn.Module):

    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9):
        super(PrimaryCaps, self).__init__()
        self.num_capsules = num_capsules
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0)
            for _ in range(num_capsules)
        ])
    
    def forward(self, x):
        # (batch_size, in_channels, height, width)
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), -1, self.num_capsules)
        # (batch_size, out_dims, num_caps)
        return self.squash(u)
    
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm *  input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor
    

class DigitCaps(nn.Module):
    def __init__(self, num_capsules=10, num_routes=32*6*6, in_channels=8, out_channels=16):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if USE_CUDA:
            b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)
            
            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)
    
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm *  input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor
    
class Decoder(nn.Module):

    def __init__(self, hidden_size=16*10, output_size=(1, 28, 28), num_classes=10):

        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.output_size = output_size
        self.reconstraction_layers = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.output_size[0] * self.output_size[1] * self.output_size[2]),
            nn.Tanh() # from -1 to 1
        )
        
    def forward(self, x, data):
        classes = torch.sqrt((x ** 2).sum(2))
        classes = F.softmax(classes)
        max_values, max_indices = classes.max(dim=1)

        masked = Variable(torch.sparse.torch.eye(self.num_classes))
        if USE_CUDA:
            masked = masked.cuda()
        # masked = (batch_size, num_classes) 0/1, 1 is highest probability class
        masked = masked.index_select(dim=0, index=max_indices.squeeze(1).data)
        x = (x * masked[:, :, None, None]).view(x.size(0), -1)
        reconstructions = self.reconstraction_layers(x)
        reconstructions = reconstructions.view(-1, self.output_size[0], self.output_size[1], self.output_size[2])
        
        return reconstructions, masked
    

padding_size = 200

test_data_0 = torch.rand((5, 768, padding_size, 1))

test_layer_0 = ConvLayer(in_channels=768, out_channels=256, kernel_size=(9,1))
test_output_0 = test_layer_0(test_data_0)
print(test_output_0.shape)

test_layer_1 = PrimaryCaps(num_capsules=8, in_channels=256, out_channels=32, kernel_size=(9, 1))
test_output_1 = test_layer_1(test_output_0)
print(test_output_1.shape)

test_layer_2 = DigitCaps(num_routes=2944, in_channels=8, num_capsules=3, out_channels=16)
test_output_2 = test_layer_2(test_output_1)
print(test_output_2.shape)

test_layer_3 = Decoder(hidden_size=3*16, output_size=(768, 200, 1), num_classes=3)
test_output_31, test_output_32 = test_layer_3(test_output_2, test_data_0)
print(test_output_31.shape, test_output_31.shape)

# caps = CapsNet()
# caps(test_data_1)


# print("Preprocessing")
# # Load the tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

# X_train = train_df['free_text'].fillna('')
# y_train = train_df['label_id'].values

# X_dev = dev_df['free_text'].fillna('')
# y_dev = dev_df['label_id'].values

# X_test = test_df['free_text'].fillna('')
# y_test = test_df['label_id'].values

# X_train_ids, X_train_attn_masks = full_preprocess(X_train, tokenizer)
# X_dev_ids, X_dev_attn_masks = full_preprocess(X_dev, tokenizer)
# X_test_ids, X_test_attn_masks = full_preprocess(X_test, tokenizer)

# # Create datasets
# train_dataset = TextDataset(X_train_ids, X_train_attn_masks, y_train)
# dev_dataset = TextDataset(X_dev_ids, X_dev_attn_masks, y_dev)
# test_dataset = TextDataset(X_test_ids, X_test_attn_masks, y_test)

# print("Creating dataloader")
# # Create dataloaders
# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
# test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)