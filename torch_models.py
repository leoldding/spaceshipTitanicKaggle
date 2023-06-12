#%% Imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

#%% Data Processing

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


deck, num, side = [], [], []

for cabin in train['Cabin']:
    if not pd.isna(cabin):
        split = cabin.split('/')
        deck.append(split[0])
        num.append(int(split[1]))
        side.append(split[2])
    else:
        deck.append(cabin)
        num.append(cabin)
        side.append(cabin)

train['Deck'] = deck
train['Num'] = num
train['Side'] = side

group_size = {pid.split('_')[0]: int(pid.split('_')[1]) for pid in train['PassengerId']}
train['GroupSize'] = [group_size[pid.split('_')[0]] for pid in train['PassengerId']]

deck, num, side = [], [], []

for cabin in test['Cabin']:
    if not pd.isna(cabin):
        split = cabin.split('/')
        deck.append(split[0])
        num.append(int(split[1]))
        side.append(split[2])
    else:
        deck.append(cabin)
        num.append(cabin)
        side.append(cabin)

test['Deck'] = deck
test['Num'] = num
test['Side'] = side

group_size = {pid.split('_')[0]: int(pid.split('_')[1]) for pid in test['PassengerId']}
test['GroupSize'] = [group_size[pid.split('_')[0]] for pid in test['PassengerId']]


train.drop(['PassengerId', 'Name', 'Cabin'], axis=1, inplace=True)
test_passenger_id = test['PassengerId']
test.drop(['PassengerId', 'Name', 'Cabin'], axis=1, inplace=True)

train, val = train_test_split(train, train_size=0.8, random_state=42)


numeric_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Num', 'GroupSize']

# log transform highly skewed columns
cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'GroupSize']

for column in cols:
    replace = np.log(np.nanmedian(train[column])+1)
    train[column] = [np.log(value+1) if not pd.isna(value) else replace for value in train[column]]
    val[column] = [np.log(value+1) if not pd.isna(value) else replace for value in val[column]]
    test[column] = [np.log(value+1) if not pd.isna(value) else replace for value in test[column]]

# square root transform Num column
replace = np.sqrt(np.nanmedian(train['Num']))
train['Num'] = [np.sqrt(value) if not pd.isna(value) else replace for value in train['Num']]
val['Num'] = [np.sqrt(value) if not pd.isna(value) else replace for value in val['Num']]
test['Num'] = [np.sqrt(value) if not pd.isna(value) else replace for value in test['Num']]

# random normal imputation for Age column
age_mean = np.nanmean(train['Age'])
age_std = np.nanstd(train['Age'])
train['Age'] = [value if not pd.isna(value) else np.random.normal(age_mean, age_std) for value in train['Age']]
val['Age'] = [value if not pd.isna(value) else np.random.normal(age_mean, age_std) for value in val['Age']]
test['Age'] = [value if not pd.isna(value) else np.random.normal(age_mean, age_std) for value in test['Age']]

# scale numeric columns
scaler = StandardScaler()
for column in numeric_cols:
    train[column] = scaler.fit_transform(np.array(train[column]).reshape(-1, 1))
    val[column] = scaler.transform(np.array(val[column]).reshape(-1, 1))
    test[column] = scaler.transform(np.array(test[column]).reshape(-1, 1))


categorical_columns = ['CryoSleep', 'VIP', 'Transported', 'Side', 'HomePlanet', 'Destination', 'Deck']

# probability imputation
for column in categorical_columns:
    counts = train[column].value_counts()
    probabilities = counts / sum(counts)

    train[column] = [value if not pd.isna(value) else np.random.choice(probabilities.index, p=probabilities) for value in train[column]]
    val[column] = [value if not pd.isna(value) else np.random.choice(probabilities.index, p=probabilities) for value in val[column]]
    if column != 'Transported':
        test[column] = [value if not pd.isna(value) else np.random.choice(probabilities.index, p=probabilities) for value in test[column]]

binary_cols = ['CryoSleep', 'VIP', 'Transported', 'Side']

le = LabelEncoder()
for column in binary_cols:
    train[column] = le.fit_transform(train[column])
    val[column] = le.transform(val[column])
    if column != 'Transported':
        test[column] = le.transform(test[column])


string_cols = ['HomePlanet', 'Destination', 'Deck']

for column in string_cols:
    encoding = pd.get_dummies(train[column])
    if 'Mars' in encoding.columns:
        encoding.drop(['Mars'], axis=1, inplace=True)
    if 'PSO J318.5-22' in encoding.columns:
        encoding.drop(['PSO J318.5-22'], axis=1, inplace=True)
    if 'T' in encoding.columns:
        encoding.drop(['T'], axis=1, inplace=True)
    train = pd.concat([train, encoding], axis=1)
    train.drop(column, axis=1, inplace=True)

    encoding = pd.get_dummies(val[column])
    if 'Mars' in encoding.columns:
        encoding.drop(['Mars'], axis=1, inplace=True)
    if 'PSO J318.5-22' in encoding.columns:
        encoding.drop(['PSO J318.5-22'], axis=1, inplace=True)
    if 'T' in encoding.columns:
        encoding.drop(['T'], axis=1, inplace=True)
    val = pd.concat([val, encoding], axis=1)
    val.drop(column, axis=1, inplace=True)

    encoding = pd.get_dummies(test[column])
    if 'Mars' in encoding.columns:
        encoding.drop(['Mars'], axis=1, inplace=True)
    if 'PSO J318.5-22' in encoding.columns:
        encoding.drop(['PSO J318.5-22'], axis=1, inplace=True)
    if 'T' in encoding.columns:
        encoding.drop(['T'], axis=1, inplace=True)
    test = pd.concat([test, encoding], axis=1)
    test.drop(column, axis=1, inplace=True)

x_train = train.drop(['Transported'], axis=1)
y_train = train['Transported']
train_dataset = TensorDataset(torch.tensor(x_train.values, dtype=torch.float32),
                              torch.tensor(y_train.values, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

x_val = val.drop(['Transported'], axis=1)
y_val = val['Transported']
val_dataset = TensorDataset(torch.tensor(x_val.values, dtype=torch.float32),
                            torch.tensor(y_val.values, dtype=torch.long))
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

#%% Model Training


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.linear_1 = nn.Linear(22, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, 32)
        self.linear_4 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_3(x)
        x = self.relu(x)
        x = self.linear_4(x)
        x = self.softmax(x)
        return x


class LossMeter:
    def __init__(self):
        self.average = 0
        self.count = 0
        self.sum = 0
        self.reset()

    # return all values to 0
    def reset(self):
        self.average = 0
        self.sum = 0
        self.count = 0

    # calculate new values
    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.average = self.sum / self.count


model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
loss_meter = LossMeter()
train_loss = []
val_loss = []
min_val_loss = float('inf')
best_model_params = model.state_dict()

for epoch in range(1, 51):
    print(f'Epoch {epoch}:')

    model.train()
    loss_meter.reset()
    tqdm_train = tqdm(train_loader, total=len(train_loader), desc='Train')
    for inputs, labels in tqdm_train:
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), len(labels))
        tqdm_train.set_postfix(loss=loss_meter.average)
    train_loss.append(loss_meter.average)

    model.eval()
    loss_meter.reset()
    tqdm_val = tqdm(val_loader, total=len(val_loader), desc='Validation')
    for inputs, labels in tqdm_val:
        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss_meter.update(loss.item(), len(labels))
        tqdm_val.set_postfix(loss=loss_meter.average)
    val_loss.append(loss_meter.average)

    if loss_meter.average < min_val_loss:
        min_val_loss = loss_meter.average
        best_model_params = model.state_dict()

plt.plot(range(1, 51), train_loss, label='Training Loss')
plt.plot(range(1, 51), val_loss, label='Validation Loss')
plt.title('Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot')
plt.close()

#%% Prediction

model.load_state_dict(best_model_params)

inputs = torch.tensor(test.values, dtype=torch.float32)

outputs = model(inputs)

predictions = [True if np.argmax(output) == 1 else False for output in outputs.detach().numpy()]

submission = pd.DataFrame(list(zip(test_passenger_id, predictions)), columns=['PassengerId', 'Transported'])
submission.to_csv('predictions.csv', index=False)
