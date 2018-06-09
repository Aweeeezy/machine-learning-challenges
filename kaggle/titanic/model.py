from __future__ import print_function

import pandas as pd

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.preprocessing import minmax_scale, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)



###############################################################################
#       Preprocess datasets
###############################################################################


# Read in train and test sets
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Remove unnecessary column(s)
train = train.drop(columns=['Name'])
test = test.drop(columns=['Name'])

# Impute missing CABIN values with some arbitrary categorical label
train['Cabin'] = train['Cabin'].fillna('-1')
test['Cabin'] = test['Cabin'].fillna('-1')

# Encode CABIN values as categorical labels
le = LabelEncoder()
train['Cabin'] = le.fit_transform(train['Cabin'])
test['Cabin'] = le.fit_transform(test['Cabin'])

# Encode TICKET values as categorical labels
le = LabelEncoder()
train['Ticket'] = le.fit_transform(train['Ticket'])
test['Ticket'] = le.fit_transform(test['Ticket'])

# Impute missing AGE values in train and test set using training mean
avg_age = round(train['Age'].mean(), 1)
train['Age'] = train['Age'].fillna(avg_age)
test['Age'] = test['Age'].fillna(avg_age)

# Impute missing FARE values in the test set using training mean
test['Fare'] = test['Fare'].fillna(round(train['Fare'].mean(), 4))

# Impute missing EMBARKED values in the train set using mode
train['Embarked'] = train['Embarked'].fillna('S')

# Replace EMBARKED string values with categorical integers
train.loc[train['Embarked'] == 'S', 'Embarked'] = 0
train.loc[train['Embarked'] == 'Q', 'Embarked'] = 1
train.loc[train['Embarked'] == 'C', 'Embarked'] = 2
test.loc[test['Embarked'] == 'S', 'Embarked'] = 0
test.loc[test['Embarked'] == 'Q', 'Embarked'] = 1
test.loc[test['Embarked'] == 'C', 'Embarked'] = 2

# Replace EMBARKED string values with categorical integers
train.loc[train['Sex'] == 'male', 'Sex'] = 0
train.loc[train['Sex'] == 'female', 'Sex'] = 1
test.loc[test['Sex'] == 'male', 'Sex'] = 0
test.loc[test['Sex'] == 'female', 'Sex'] = 1

# Normalize non-categorical features
for feature in ['SibSp', 'Parch', 'Fare', 'Age', 'Ticket', 'Cabin']:
    train[feature] = minmax_scale(train[feature])
    test[feature] = minmax_scale(test[feature])

# Isolate training labels from data
train_y = train['Survived'].values
train = train.drop(columns=['Survived', 'PassengerId'])
train_x = train.values
test_Ids = test['PassengerId'].values
test = test.drop(columns=['PassengerId'])
test_x = test.values


###############################################################################
#       Train model(s)
###############################################################################


models = [
            (GaussianNB, {}),
            (DecisionTreeClassifier, {'criterion': ['gini', 'entropy']}),
            (LogisticRegression,
                    {'C': [0.001, 0.01, 0.1, 0.5, 1, 1.5, 2, 10]}),
            (KNeighborsClassifier, {'n_neighbors': [3,4,5,6,7,8,9,10]}),
            (SVC, [{'kernel': ['linear'],
                      'C': [0.001, 0.01, 0.1, 0.5, 1, 1.5, 2, 10]},
                     {'kernel': ['rbf'],
                      'C': [0.001, 0.01, 0.1, 0.5, 1, 1.5, 2, 10],
                      'gamma': ['auto', 0.001, 0.01, 0.1, 0.5, 1, 1.5, 2, 10]}
                     ]),
            (RandomForestClassifier,
                    {'criterion': ['gini', 'entropy'],
                     'n_estimators': [3,4,5,6,7,8,9,10],
                     'max_depth': [None,2,3,4,5,6,7,8,9,10]}),
            (XGBClassifier, {'max_depth': [3,4,5,6,7,8,9,10],
                             'n_estimators': [100, 101, 102, 103, 104],
                             'gamma': [0.098, 0.1, 0.102],
                             'learning_rate': [0.099, 0.1, 0.101]})
          ]

trained = []

for model, params in models:

    if type(model()).__name__ in ['GaussianNB', 'KNeighborsClassifier']:
        m = model()
    else:
        m = model(random_state=42)

    grid = GridSearchCV(estimator=m, param_grid=params, cv=10, n_jobs=-1)
    grid.fit(train_x, train_y)
    score = grid.score(train_x, train_y)
    pred = grid.predict(train_x)
    print("Grid:\n", grid)
    print("Training score:", score)
    print("Best cross-validation score:", grid.best_score_)
    print("CV/train difference:", score - grid.best_score_)
    print("Best params:", grid.best_params_)
    print("Classification report:\n{}\n\n".format(classification_report(train_y, pred)))

    trained.append((grid.best_score_, grid.best_estimator_))

best_model = sorted(trained, reverse=True)[0][1]
print("Best model is", type(best_model).__name__, "with a cross-validation score of", sorted(trained, reverse=True)[0][0])
predictions = best_model.predict(test_x)

with open('data/submission.csv', 'w') as f:
    f.write('PassengerId,Survived\n')
    for passenger, prediction in zip(test_Ids, predictions):
        f.write("{},{}\n".format(passenger, prediction))
