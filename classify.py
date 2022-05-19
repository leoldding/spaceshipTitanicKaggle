#%% Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

#%% Data
data = pd.read_csv('train.csv')
data.drop(['PassengerId', 'Name', 'Destination', 'HomePlanet'], axis=1, inplace=True)
tf = ['CryoSleep', 'VIP', 'Transported']
for col in tf:
    data[col] = [1 if x else 0 for x in data[col]]
deck = []
num = []
side = []
for i in data['Cabin']:
    temp = i.split('/')
    deck.append(temp[0])
    num.append(int(temp[1]))
    side.append(temp[2])
data.drop('Cabin', axis=1, inplace=True)
data['Deck'] = deck
data['Num'] = num
data['Side'] = side

deckOrder = data['Deck'].unique()
deckOrder.sort()

deckVals = {}
count = 0
for i in range(len(deckOrder)):
    deckVals[deckOrder[i]] = count
    count += 1

data['Deck'] = [deckVals[x] for x in data['Deck']]
data['Side'] = [1 if x == 'P' else 0 for x in data['Side']]

y = data['Transported']
x = data.drop('Transported', axis=1)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.85, random_state=42)

#%% Models
# Logistic Regression

lr = LogisticRegression(max_iter=1000, random_state=42).fit(xtrain, ytrain)
lrpred = lr.predict(xtest)
lrroc = roc_auc_score(lrpred, ytest)
print('Logistic Regression ROC: ', lrroc)

# Random Forest

rf = RandomForestClassifier(random_state=42).fit(xtrain, ytrain)
rfpred = rf.predict(xtest)
rfroc = roc_auc_score(rfpred, ytest)
print('Random Forest ROC: ', rfroc)

# AdaBoost

ab = AdaBoostClassifier(random_state=42).fit(xtrain, ytrain)
abpred = ab.predict(xtest)
abroc = roc_auc_score(abpred, ytest)
print('ADABoost ROC: ', abroc)

# Naive Bays

nb = GaussianNB().fit(xtrain, ytrain)
nbpred = nb.predict(xtest)
nbroc = roc_auc_score(nbpred, ytest)
print('Naive Bayes ROC: ', nbroc)

# XGBoost

xgb = xgboost.XGBClassifier().fit(xtrain, ytrain)
xgbpred = xgb.predict(xtest)
xgbroc = roc_auc_score(xgbpred, ytest)
print('XGBoost ROC: ', xgbroc)

#%% Hyperparameter Tuning

parameters = [{'n_estimators': range(100, 550, 50), 'criterion': ['gini', 'entropy']}]

gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=parameters, scoring='roc_auc', cv=5, verbose=3)
gs.fit(xtrain, ytrain)
print(gs.best_params_)
print(gs.best_score_)

#%%
bestModel = RandomForestClassifier(random_state=42, criterion='entropy', n_estimators=400)
bestModel.fit(xtrain, ytrain)


