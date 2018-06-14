from __future__ import print_function
import sys

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, MultiTaskLasso, ElasticNet, MultiTaskElasticNet, SGDRegressor

from sklearn.preprocessing import minmax_scale, LabelEncoder
from sklearn.model_selection import GridSearchCV

import warnings
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)



###############################################################################
###############################################################################
#       Data Preprocessing
###############################################################################
###############################################################################



# Read in train and test sets
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Isolate labels and remove unneeded features
#train_y = train['SalePrice'].values
#train = train.drop(columns=['SalePrice', 'Id'])
train = train.drop(columns=['Id'])
test_IDs = test['Id'].values
test = test.drop(columns=['Id'])


###############################################################################
#       Imputation
###############################################################################


# Impute LOTFRONTAGE with 0
train['LotFrontage'] = train['LotFrontage'].fillna(0)
test['LotFrontage'] = test['LotFrontage'].fillna(0)

# Impute ALLEY with NA
train['Alley'] = train['Alley'].fillna('NA')
test['Alley'] = test['Alley'].fillna('NA')

# Impute MASVNRTYPE with train mode
mode = train['MasVnrType'].mode().iloc[0]
train['MasVnrType'] = train['MasVnrType'].fillna(mode)
test['MasVnrType'] = test['MasVnrType'].fillna(mode)

# Impute MASVNRAREA with train mean
mean = train['MasVnrArea'].mean()
train['MasVnrArea'] = train['MasVnrArea'].fillna(mean)
test['MasVnrArea'] = test['MasVnrArea'].fillna(mean)

# Impute BSMTQUAL with NA
train['BsmtQual'] = train['BsmtQual'].fillna('NA')
test['BsmtQual'] = test['BsmtQual'].fillna('NA')

# Impute BSMTCOND with NA
train['BsmtCond'] = train['BsmtCond'].fillna('NA')
test['BsmtCond'] = test['BsmtCond'].fillna('NA')

# Impute BSMTEXPOSURE with NA
train['BsmtExposure'] = train['BsmtExposure'].fillna('NA')
test['BsmtExposure'] = test['BsmtExposure'].fillna('NA')

# Impute BSMTFINTYPE1 with NA
train['BsmtFinType1'] = train['BsmtFinType1'].fillna('NA')
test['BsmtFinType1'] = test['BsmtFinType1'].fillna('NA')

# Impute BSMTFINTYPE2 with NA
train['BsmtFinType2'] = train['BsmtFinType2'].fillna('NA')
test['BsmtFinType2'] = test['BsmtFinType2'].fillna('NA')

# Impute ELECTRICAL with training mode
train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode().iloc[0])

# Impute FIREPLACEQU with NA
train['FireplaceQu'] = train['FireplaceQu'].fillna('NA')
test['FireplaceQu'] = test['FireplaceQu'].fillna('NA')

# Impute GARAGETYPE with NA
train['GarageType'] = train['GarageType'].fillna('NA')
test['GarageType'] = test['GarageType'].fillna('NA')

# Impute GARAGEYRBLT with 0
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(0)
test['GarageYrBlt'] = test['GarageYrBlt'].fillna(0)

# Impute GARAGEFINISH with NA
train['GarageFinish'] = train['GarageFinish'].fillna('NA')
test['GarageFinish'] = test['GarageFinish'].fillna('NA')

# Impute GARAGEQUAL with NA
train['GarageQual'] = train['GarageQual'].fillna('NA')
test['GarageQual'] = test['GarageQual'].fillna('NA')

# Impute GARAGECOND with NA
train['GarageCond'] = train['GarageCond'].fillna('NA')
test['GarageCond'] = test['GarageCond'].fillna('NA')

# Impute POOLQC with NA
train['PoolQC'] = train['PoolQC'].fillna('NA')
test['PoolQC'] = test['PoolQC'].fillna('NA')

# Impute FENCE with NA
train['Fence'] = train['Fence'].fillna('NA')
test['Fence'] = test['Fence'].fillna('NA')

# Impute MISCFEATURE with NA
train['MiscFeature'] = train['MiscFeature'].fillna('NA')
test['MiscFeature'] = test['MiscFeature'].fillna('NA')

# Impute MSZONING with training mode
test['MSZoning'] = test['MSZoning'].fillna(train['MSZoning'].mode().iloc[0])

# Impute UTILITIES with training mode
test['Utilities'] = test['Utilities'].fillna(train['Utilities'].mode().iloc[0])

# Impute EXTERIOR1ST with training mode
test['Exterior1st'] = test['Exterior1st'].fillna(train['Exterior1st'].mode().iloc[0])

# Impute EXTERIOR2ND with training mode
test['Exterior2nd'] = test['Exterior2nd'].fillna(train['Exterior2nd'].mode().iloc[0])

# Impute BSMTFINSF1 with training mean
test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(train['BsmtFinSF1'].mean())

# Impute BSMTFINSF2 with training mean
test['BsmtFinSF2'] = test['BsmtFinSF2'].fillna(train['BsmtFinSF2'].mean())

# Impute BSMTUNFSF with training mean
test['BsmtUnfSF'] = test['BsmtUnfSF'].fillna(train['BsmtUnfSF'].mean())

# Impute TOTALBSMTSF with training mean
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(train['TotalBsmtSF'].mean())

# Impute BSMTFULLBATH with training mode
test['BsmtFullBath'] = test['BsmtFullBath'].fillna(0)

# Impute BSMTHALFBATH with training mode
test['BsmtHalfBath'] = test['BsmtHalfBath'].fillna(0)

# Impute KITCHENQUAL with training mode
test['KitchenQual'] = test['KitchenQual'].fillna(train['KitchenQual'].mode().iloc[0])

# Impute FUNCTIONAL with training mode
test['Functional'] = test['Functional'].fillna(train['Functional'].mode().iloc[0])

# Impute GARAGECARS with training mode
test['GarageCars'] = test['GarageCars'].fillna(1)

# Impute GARAGEAREA with training mean
test['GarageArea'] = test['GarageArea'].fillna(train['GarageArea'].mean())

# Impute SALETYPE with training mode
test['SaleType'] = test['SaleType'].fillna(train['SaleType'].mode().iloc[0])


###############################################################################
#       Label Encoding
###############################################################################


# Select numerical features for outlier analysis before encoding categorical
# features
x = train.select_dtypes(include=['int', 'float'])

to_encode = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape',
        'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'YearBuilt',
        'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
        'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
        'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating',
        'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
        'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
        'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'MoSold',
        'SaleType', 'SaleCondition']

for feature in to_encode:
    le = LabelEncoder()
    train[feature] = le.fit_transform(train[feature])
    test[feature] = le.fit_transform(test[feature])


###############################################################################
#       Feature Scaling
###############################################################################


for feature in train.columns.tolist():
    if feature not in ['OverallQual', 'OverallCond', 'SalePrice']:
        train[feature] = minmax_scale(train[feature])
        test[feature] = minmax_scale(test[feature])


###############################################################################
#       Remove Outliers
###############################################################################

dataset_names = ['4th std.']

# Datasets w/ values 4 standard deviations outside the mean marked
datasets = [
                train[(np.abs(x - x.mean()) <= (4 * x.std())).all(axis=1)],
        ]

# Seperate train_x and train_y
datasets = [(d.drop(columns=['SalePrice']), d['SalePrice'], name)
        for d, name in zip(datasets, dataset_names)]



###############################################################################
###############################################################################
#       Model Training
###############################################################################
###############################################################################



# Tried -- they suck
crap_models = [
            (SVR, [{'C': [0.01, 0.1, 0.5, 1, 2],
                    'epsilon': [0.01, 0.1, 0.5, 1, 2],
                    'kernel': ['rbf'],
                    'gamma': [0.01, 0.1, 0.5, 1, 2]},
                   {'C': [0.01, 0.1, 0.5, 1, 2],
                    'epsilon': [0.01, 0.1, 0.5, 1, 2],
                    'kernel': ['linear']}]),
            (NuSVR, [{'C': [0.01, 0.1, 0.5, 1, 2],
                      'kernel': ['rbf'],
                      'gamma': [0.01, 0.1, 0.5, 1, 2],
                      'nu': [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]},
                     {'C': [0.01, 0.1, 0.5, 1, 2],
                      'kernel': ['linear'],
                      'nu': [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]}]),
            (LinearRegression, {}),
            (DecisionTreeRegressor, {'criterion': ['mse', 'friedman_mse', 'mae']}),
            (KNeighborsRegressor, {'n_neighbors': [2,3,4,5,6,7,8,9,10]}),
            (ElasticNet, {'alpha': [0.00380],
                          'l1_ratio': [0.06]})
        ]

# Good, but not good enough
proven = [
            (RandomForestRegressor,
                    {'criterion': ['mae'],
                     'n_estimators': [8,10,12,14,16,18,20],
                     'max_depth': [None,2,4,6,8,10,12,14,16]}),
            (SGDRegressor, {'loss': ['squared_epsilon_insensitive'],
                            'penalty': ['none'],
                            'alpha': [0.0000001, 0.0000005, 0.0000009, 0.000001, 0.000002, 0.000003],
                            'l1_ratio': [0.00007, 0.00008, 0.00009, 0.000095]}),
            (Ridge, {'alpha': [4.6927]}),
            (Lasso, {'alpha': [106]})
        ]

models = [
            (XGBRegressor, {'max_depth': [4],
                            'n_estimators': [1486],
                            'learning_rate': [0.1],
                            'gamma': [0.00000000001]})
        ]

trained = []

no_random_state = ['SVR', 'NuSVR', 'LinearRegression', 'KNeighborsRegressor',
                   'DecisionTreeRegressor']

print("Training {} model(s) on {} dataset(s):".format(len(models), len(datasets)))

for train_x, train_y, name in datasets:

    print("\t", name, "...\n")

    for model, params in models:
        if type(model()).__name__ in no_random_state:
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
        print("Best params: {}\n".format(grid.best_params_))

        trained.append((grid.best_score_, grid.best_estimator_))

best_model = sorted(trained, reverse=True)[0][1]
print("Best model is", type(best_model).__name__, "with a cross-validation score of", sorted(trained,     reverse=True)[0][0])
predictions = best_model.predict(test)

with open('data/submission.csv', 'w') as f:
    f.write('Id,SalePrice\n')
    for house, prediction in zip(test_IDs, predictions):
        f.write("{},{}\n".format(house, prediction))
