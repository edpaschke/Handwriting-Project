import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("nibinv23/iam-handwriting-word-database")
print("Path to dataset files:", path)

# Check the structure so you know what's inside
for root, dirs, files in os.walk(path):
    level = root.replace(path, '').count(os.sep)
    if level < 3:  # only print top 3 levels
        print(' ' * 2 * level + os.path.basename(root) + '/')
        
      
chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
num_classes = len(chars) + 1
char_to_idx = {c:i+1 for i,c in enumerate(chars)}
idx_to_char = {i+1:c for c,i in char_to_idx.items()}

# Dataset
class IAM_Dataset(Dataset):
    def __init__(self, words_txt_path, img_dir):
        self.samples = []
        self.img_dir = img_dir
        with open(words_txt_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                if parts[1] != 'ok':
                    continue

                word_id = parts[0]
                transcription = parts[-1]

                split = word_id.split('-')
                folder1 = split[0]
                folder2 = f'{split[0]}-{split[1]}'

                img_path = f'{img_dir}/{folder1}/{folder2}/{word_id}.png'
                self.samples.append((img_path, transcription))

    def encode_text(self, text):
        text = text.lower().strip()
        cleaned = []
        for c in text:
            if c in char_to_idx:
                cleaned.append(char_to_idx[c])

        return cleaned

    def preprocess(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        height = 32
        width = int(img.shape[1] * (32 / img.shape[0]))
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0).float()
        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]

        img = self.preprocess(img_path)
        if img is None:
            return self.__getitem__((idx + 1) % len(self.samples))
        label = torch.tensor(self.encode_text(text), dtype=torch.long)

        return img, label

# Collate Function
def collate(batch):
    images = []
    labels = []
    label_lengths = []
    widths = []

    for img, label in batch:
        images.append(img)
        labels.append(label)
        label_lengths.append(len(label))
        widths.append(img.shape[2])

    max_width = max(widths)

    padded_images = []
    for img in images:
        c, h, w = img.shape
        pad_width = max_width - w
        padded = F.pad(img, (0, pad_width), value=0)
        padded_images.append(padded)

    images = torch.stack(padded_images, dim=0)
    labels = torch.cat(labels)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)

    return images, labels, label_lengths

# Feature Extractor
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 1),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 1),
        )

    def forward(self, x):
        x = self.features(x)

        x = x.mean(dim=2)
        seq_len = x.shape[-1]

        return x, seq_len

# CRNN Model
class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNN()

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=512,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x, seq_len = self.cnn(x)

        x = x.permute(0, 2, 1)

        x, _ = self.lstm(x)
        x = self.fc(x)

        return x, seq_len

# Setup
learning_rate = 1e-3
ctc_loss = nn.CTCLoss(blank=0)
model = CRNN()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = model.to(device)

# Dataloader

words_txt = os.path.join(path, 'iam_words', 'words.txt')
img_dir   = os.path.join(path, 'iam_words', 'words')

dataset = IAM_Dataset(words_txt, img_dir)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# # TEMP
# train_dataset = torch.utils.data.Subset(train_dataset, range(20000))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate)

# Decode
def decode(pred):
    pred = pred.detach().cpu().argmax(2)
    result = []
    for seq in pred:
        text = []
        prev = -1
        for p in seq:
            idx = p.item()
            if idx != prev and idx != 0:
                text.append(idx_to_char.get(idx, ''))
            prev = idx
        result.append(''.join(text))
    return result

# Training
model.train()

for epoch in range(10):
    print(f'\nEpoch {epoch+1}')
    epoch_loss = 0

    for images, labels, label_lengths in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        optimizer.zero_grad()

        preds, seq_len = model(images)
        preds = preds.log_softmax(2)

        B = preds.size(0)

        input_lengths = torch.full(
            (B,),
            seq_len,
            dtype=torch.long,
            device=device
        )

        loss = ctc_loss(
            preds.permute(1, 0, 2),
            labels,
            input_lengths,
            label_lengths
        )

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f'Loss: {epoch_loss / len(train_loader)}')

# Testing
model.eval()

all_preds = []
all_targets = []

with torch.no_grad():
    for images, labels, label_lengths in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        preds, _ = model(images)
        preds = preds.log_softmax(2)

        decoded = decode(preds)
        all_preds.extend(decoded)

        start = 0
        batch_targets = []
        for length in label_lengths:
            target_seq = labels[start:start+length]
            text = ''.join([idx_to_char.get(i.item(), '') for i in target_seq])
            batch_targets.append(text)
            start += length

        all_targets.extend(batch_targets)

print('\nSample Predictions:')
for i in range(5):
    print(f'PRED: {all_preds[i]}')
    print(f'TRUE: {all_targets[i]}')
    print()
