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

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print('use_cuda: ', USE_CUDA)
print('cuda: ', device)

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
import torch.nn.functional as F

class ConvLayer(nn.Module):

    def __init__(self, in_channels=1, out_channels=256, kernel_size=9, stride=2):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1)

    def forward(self, x):
        return F.relu(self.conv(x))



class PrimaryCaps(nn.Module):

    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9):
        super(PrimaryCaps, self).__init__()
        self.num_capsules = num_capsules
        self.capsules = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0)
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
    
class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()

        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
        self.phobert.train()
        self.phobert.requires_grad_(True)

        self.conv_layer = ConvLayer(in_channels=1, out_channels=128, kernel_size=512, stride=128)
        self.primary_capsules = PrimaryCaps(num_capsules=8, in_channels=128, out_channels=32, kernel_size=9)
        self.digit_capsules = DigitCaps(num_capsules=32, num_routes=4000, in_channels=8, out_channels=32)
        
        self.FCs = nn.Sequential(
            nn.Linear(32*32, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, ids, attn_mask):
        x = self.phobert(ids, attn_mask).last_hidden_state.mean(dim=1).unsqueeze(1)
        x = self.digit_capsules(self.primary_capsules(self.conv_layer(x)))
        x = x.reshape(x.shape[0], -1)
        x = self.FCs(x)
        soft_max_output = F.log_softmax(x, dim=1)
        return soft_max_output


print("Preprocessing")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

X_train = train_df['free_text'].fillna('')
y_train = train_df['label_id'].values

X_dev = dev_df['free_text'].fillna('')
y_dev = dev_df['label_id'].values

X_test = test_df['free_text'].fillna('')
y_test = test_df['label_id'].values

X_train_ids, X_train_attn_masks = full_preprocess(X_train, tokenizer)
X_dev_ids, X_dev_attn_masks = full_preprocess(X_dev, tokenizer)
X_test_ids, X_test_attn_masks = full_preprocess(X_test, tokenizer)

# Create datasets
train_dataset = TextDataset(X_train_ids, X_train_attn_masks, y_train)
dev_dataset = TextDataset(X_dev_ids, X_dev_attn_masks, y_dev)
test_dataset = TextDataset(X_test_ids, X_test_attn_masks, y_test)

print("Creating dataloader")
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

capsule_net = CapsNet().to('cuda')
print(capsule_net)
from torch.optim import Adam
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(capsule_net.parameters(), lr=0.001)

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
epochs = 10
for epoch in range(epochs):
    capsule_net.train()
    train_loss = 0
    true_labels = []
    predictions = []

    with tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}") as tqdm_loader:
        for batch_idx, batch in enumerate(tqdm_loader):
            tqdm_loader.set_description(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_dataloader)}")

            ids = batch['input_ids'].to('cuda')
            attn_mask = batch['attention_mask'].to('cuda')
            labels = batch['labels']
            target = torch.nn.functional.one_hot(labels, num_classes=3).to('cuda')

            optimizer.zero_grad()
            outputs = capsule_net(ids, attn_mask).to('cuda')
            loss = capsule_net.loss(outputs, target)
            loss.backward()
            optimizer.step()

            train_loss += loss
            
            true_labels.extend(labels.cpu().numpy().tolist())
            predictions.extend(torch.max(outputs, dim=1)[1].cpu().numpy().tolist())

    print(f'Loss: {train_loss:.4f}')
    print('Accuracy: ', accuracy_score(true_labels, predictions))
    print('F1:', f1_score(true_labels, predictions, average='macro'))


    # Evaluating
    capsule_net.eval()
    dev_loss = 0
    true_labels = []
    predictions = []
    with torch.no_grad():
        # Wrap the dataloader with tqdm to monitor progress
        with tqdm(dev_dataloader, desc="Evaluation") as tqdm_loader:
            for batch_idx, batch in enumerate(tqdm_loader):
                tqdm_loader.set_description(f"Evaluation, Batch {batch_idx + 1}/{len(dev_dataloader)}")

                ids = batch['input_ids'].to('cuda')
                attn_mask = batch['attention_mask'].to('cuda')
                labels = batch['labels']
                target = torch.nn.functional.one_hot(labels, num_classes=3).to('cuda')

                outputs = capsule_net(ids, attn_mask).to('cuda')
                loss = capsule_net.loss(outputs, target)
                
                dev_loss += loss
                
                true_labels.extend(labels.cpu().numpy().tolist())
                predictions.extend(torch.max(outputs, dim=1)[1].cpu().numpy().tolist())

    print(f'Loss: {dev_loss:.4f}')
    print('Accuracy: ', accuracy_score(true_labels, predictions))
    print('F1:', f1_score(true_labels, predictions, average='macro'))
    print()
    print()



print('on test')
capsule_net.eval()
test_loss = 0
true_labels = []
predictions = []
with torch.no_grad():
    # Wrap the dataloader with tqdm to monitor progress
    with tqdm(test_dataloader, desc="Evaluation") as tqdm_loader:
        for batch_idx, batch in enumerate(tqdm_loader):
            tqdm_loader.set_description(f"Evaluation, Batch {batch_idx + 1}/{len(test_dataloader)}")

            ids = batch['input_ids'].to('cuda')
            attn_mask = batch['attention_mask'].to('cuda')
            labels = batch['labels']
            target = torch.nn.functional.one_hot(labels, num_classes=3).to('cuda')

            outputs = capsule_net(ids, attn_mask).to('cuda')
            loss = capsule_net.loss(outputs, target)
            
            test_loss += loss
            
            true_labels.extend(labels.cpu().numpy().tolist())
            predictions.extend(torch.max(outputs, dim=1)[1].cpu().numpy().tolist())

print(f'Loss: {test_loss:.4f}')
print('Accuracy: ', accuracy_score(true_labels, predictions))
print('F1:', f1_score(true_labels, predictions, average='macro'))
print()
print()