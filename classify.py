#%% Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import scipy.stats as scp

#%% Train Data
data = pd.read_csv('data/train.csv')

print(data.columns)
print(data.isna().sum())
print(data.info())

drop = ['PassengerId', 'Name', 'Destination', 'HomePlanet']
data.drop(drop, axis=1, inplace=True)
data.dropna(inplace=True)

deck = []
num = []
side = []
for i in data['Cabin']:
    temp = i.split('/')
    deck.append(temp[0])
    num.append(int(temp[1]))
    side.append(temp[2])
data.drop('Cabin', axis=1, inplace=True)
data['Num'] = num
data['Side'] = [1 if x == 'P' else 0 for x in side]
deckOrder = list(set(deck))
deckOrder.sort()
deckVals = {}
count = 0
for i in range(len(deckOrder)):
    deckVals[deckOrder[i]] = count
    count += 1
data['Deck'] = [deckVals[x] for x in deck]

data['CryoSleep'] = [1 if x else 0 for x in data['CryoSleep']]
data['VIP'] = [1 if x else 0 for x in data['VIP']]

for col in data.drop('Transported', axis=1).columns:
    plt.hist(data[col])
    plt.title(col)
    plt.show()

y = data['Transported']
x = data.drop('Transported', axis=1)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.80, random_state=42)

#%% Models
# Logistic Regression

lr = LogisticRegression(max_iter=1000, random_state=42).fit(xtrain, ytrain)
lrpred = lr.predict(xtest)
lrroc = roc_auc_score(ytest, lrpred)
print('Logistic Regression ROC: ', lrroc)

# Random Forest

rf = RandomForestClassifier(random_state=42).fit(xtrain, ytrain)
rfpred = rf.predict(xtest)
rfroc = roc_auc_score(ytest, rfpred)
print('Random Forest ROC: ', rfroc)

# AdaBoost

ab = AdaBoostClassifier(random_state=42).fit(xtrain, ytrain)
abpred = ab.predict(xtest)
abroc = roc_auc_score(ytest, abpred)
print('ADABoost ROC: ', abroc)

# Naive Bays

nb = GaussianNB().fit(xtrain, ytrain)
nbpred = nb.predict(xtest)
nbroc = roc_auc_score(ytest, nbpred)
print('Naive Bayes ROC: ', nbroc)

# XGBoost

xgb = XGBClassifier(random_state=42).fit(xtrain, ytrain)
xgbpred = xgb.predict(xtest)
xgbroc = roc_auc_score(ytest, xgbpred)
print('XGBoost ROC: ', xgbroc)

#%% Hyperparameter Tuning

parameters = [{'n_estimators': range(10, 210, 10)}]

gsxgb = GridSearchCV(XGBClassifier(random_state=42), param_grid=parameters, scoring='roc_auc', cv=5, verbose=3)
gsxgb.fit(xtrain, ytrain)

gsab = GridSearchCV(AdaBoostClassifier(random_state=42), param_grid=parameters, scoring='roc_auc', cv = 5, verbose=3)
gsab.fit(xtrain, ytrain)

parameters = [{'n_estimators': range(25, 325, 25), 'criterion': ['gini', 'entropy']}]

gsrf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=parameters, scoring='roc_auc', cv=5, verbose=3)
gsrf.fit(xtrain, ytrain)

print(gsxgb.best_params_)
print(gsxgb.best_score_)
print(gsab.best_params_)
print(gsab.best_score_)
print(gsrf.best_params_)
print(gsrf.best_score_)

#%% Test Data

test = pd.read_csv('data/test.csv')
print(test.isna().sum())
ids = test['PassengerId']
test.drop(drop, axis=1, inplace=True)

modes = {}
for col in data.drop('Transported', axis=1).columns:
    modes[col] = scp.mode(data[col])

medians = {}
for col in data.drop('Transported', axis=1).columns:
    medians[col] = np.median(data[col])

deck = []
num = []
side = []
for i in test['Cabin']:
    if isinstance(i, float):
        deck.append(i)
        num.append(i)
        side.append(i)
    else:
        temp = i.split('/')
        deck.append(temp[0])
        num.append(int(temp[1]))
        side.append(temp[2])
test.drop('Cabin', axis=1, inplace=True)
test['Deck'] = deck
test['Num'] = num
test['Side'] = side

test['CryoSleep'] = [x if isinstance(x, float) else 1 if x else 0 for x in test['CryoSleep']]
test['VIP'] = [x if isinstance(x, float) else 1 if x else 0 for x in test['VIP']]
test['Side'] = [x if isinstance(x, float) else 1 if x == 'P' else 0 for x in test['Side']]
test['Deck'] = [x if isinstance(x, float) else deckVals[x] for x in test['Deck']]

for col in test.columns:
    print(col)
    test[col].fillna(modes[col][0][0], inplace=True)

#%% Predictions

xgb = XGBClassifier(random_state=42, n_estimators=30).fit(xtrain, ytrain)
predictions = xgb.predict(test)
predictions = [True if x == 1 else False for x in predictions]
output = pd.DataFrame(list(zip(ids, predictions)), columns=['PassengerId', 'Transported'])

output.to_csv('predictions.csv', index=False)
