import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from medmnist import BloodMNIST
from medmnist import INFO
from sklearn.metrics import accuracy_score
from scipy.stats import mode

# ==========================================================
# Feature Extraction
# ==========================================================

def get_features(img):

    img = img.astype(np.uint8)

    # Feature 1: Original
    f1 = img.flatten()

    # Feature 2: Edge
    edge = cv2.Canny(img,50,150)
    f2 = edge.flatten()

    # Feature 3: Threshold
    _,th = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    f3 = th.flatten()

    # Feature 4: Sobel
    sobel = cv2.Sobel(img,cv2.CV_64F,1,1,ksize=3)
    sobel = np.uint8(np.abs(sobel))
    f4 = sobel.flatten()

    return [
        f1/255.0,
        f2/255.0,
        f3/255.0,
        f4/255.0
    ]

# ==========================================================
# Dataset
# ==========================================================

class BloodDataset(Dataset):

    def __init__(self, split='train'):

        self.data = BloodMNIST(split=split,
                               download=True)

        self.X = []
        self.Y = []

        for img,label in self.data:

            img = np.array(img)

            if len(img.shape)==3:
                img = cv2.cvtColor(img,
                                   cv2.COLOR_RGB2GRAY)

            feats = get_features(img)

            self.X.append(feats)
            self.Y.append(label)

        self.Y = np.array(self.Y).squeeze()

    def __len__(self):
        return len(self.Y)

    def __getitem__(self,idx):

        feats = [torch.FloatTensor(f)
                 for f in self.X[idx]]

        y = torch.tensor(self.Y[idx],
                         dtype=torch.long)

        return feats,y

# ==========================================================
# ANN
# ==========================================================

class SmallMLP(nn.Module):

    def __init__(self,input_dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,8)
        )

    def forward(self,x):
        return self.net(x)

# ==========================================================
# Load Data
# ==========================================================

trainset = BloodDataset('train')
testset  = BloodDataset('test')

trainloader = DataLoader(trainset,
                         batch_size=128,
                         shuffle=True)

testloader = DataLoader(testset,
                        batch_size=128,
                        shuffle=False)

# ==========================================================
# Create 4 ANNs
# ==========================================================

input_dim = 28*28

models = [SmallMLP(input_dim)
          for _ in range(4)]

optimizers = [
    torch.optim.Adam(m.parameters(),
                     lr=0.001)
    for m in models
]

criterion = nn.CrossEntropyLoss()

# ==========================================================
# Train
# ==========================================================

epochs = 10

for epoch in range(epochs):

    for feats,y in trainloader:

        for i in range(4):

            optimizers[i].zero_grad()

            out = models[i](feats[i])

            loss = criterion(out,y)

            loss.backward()

            optimizers[i].step()

    print(f"Epoch {epoch+1}/{epochs}")

# ==========================================================
# Testing with Majority Voting
# ==========================================================

all_pred = []
all_true = []

with torch.no_grad():

    for feats,y in testloader:

        votes = []

        for i in range(4):

            out = models[i](feats[i])

            pred = torch.argmax(out,
                                dim=1)

            votes.append(pred.numpy())

        votes = np.array(votes)

        final_pred = mode(votes,
                          axis=0,
                          keepdims=False).mode

        all_pred.extend(final_pred)
        all_true.extend(y.numpy())

acc = accuracy_score(all_true,
                     all_pred)

print("\nFinal Voting Accuracy =",
      round(acc*100,2),"%")
