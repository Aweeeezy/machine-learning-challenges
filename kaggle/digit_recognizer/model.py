import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim



###############################################################################
###############################################################################
#       Data Preprocessing
###############################################################################
###############################################################################



batch_size = 32

class Data(Dataset):
    def __init__(self, training=True):
        self.training = training
        if self.training:
            train = pd.read_csv('./data/train.csv')
            train_y = train['label'].values.tolist()
            train_x = train.drop(columns=['label']).values.astype(float)
            train_x = minmax_scale(train_x)
            train_x = train_x.reshape(train.shape[0], 1, 28, 28)
            self.data_list = train_x
            self.label_list = train_y
        else:
            test_x = pd.read_csv('data/test.csv')
            test_x = minmax_scale(test_x.values.astype(float))
            test_x = test_x.reshape(test_x.shape[0], 1, 28, 28)
            self.data_list = test_x

    def __getitem__(self, index):
        if self.training:
            return torch.Tensor(self.data_list[index]), self.label_list[index]
        else:
            return torch.Tensor(self.data_list[index])

    def __len__(self):
        return self.data_list.shape[0]


train_set = Data()
test_set = Data(training=False)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
        num_workers=8)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
        num_workers=8)



###############################################################################
###############################################################################
#       Network Definition
###############################################################################
###############################################################################



class Net(nn.Module):

    torch.manual_seed(42)

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,
                        padding=2),
                nn.BatchNorm2d(num_features=16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2))
        self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5,
                        padding=2),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2))
        self.layer3 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,
                        padding=2),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2))
        self.layer4 = nn.Sequential(
                nn.Linear(64 * 4 * 4, 128),
                nn.BatchNorm1d(num_features=128),
                nn.ReLU())
        self.layer5 = nn.Sequential(
                nn.Linear(128, 64),
                nn.BatchNorm1d(num_features=64),
                nn.ReLU())
        self.layer6 = nn.Sequential(
                nn.Linear(64, 10),
                nn.BatchNorm1d(num_features=10),
                nn.Sigmoid())
        """
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, np.prod(list(x.size())[1:]).item())
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        """
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        """

        return x



###############################################################################
###############################################################################
#       Model Training
###############################################################################
###############################################################################



net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=8e-4)
num_epochs = 20

for epoch in range(num_epochs):

    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs = Variable(inputs)
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 50 == 49:
            print(f"[{epoch+1}, {i+1}] loss: {running_loss / 50}")
            running_loss = 0.0

print("Done training.\n")

# Alter forward pass behavior (affects BatchNorm, Dropout, etc.) for scoring
net.eval()

# Calculate training accuracy
score = 0
for inputs, labels in train_loader:
    inputs = Variable(inputs)
    labels = Variable(labels)

    output = F.softmax(net(inputs), dim=-1)
    _, predictions = torch.max(output, 1)

    for pred, label in zip(predictions, labels):
        if pred == label:
            score += 1

print("Training score:", score/len(train_set))

# Make predictions using the test set
ans = torch.LongTensor()
for test_input in test_loader:
    test_input = Variable(test_input)
    output = net(test_input)
    _, prediction = torch.max(output, 1)
    ans = torch.cat((ans, prediction), 0)

# Convert prediction Tensor into DataFrame and write to CSV
answers = pd.DataFrame(ans.numpy())
answers.columns = ['Label']
answers.insert(0, 'ImageId', range(1, answers.size+1))
answers.to_csv('./data/submission.csv', index=False)
