Missing values
===============

Train
------------

* LotFrontage
* Alley
* MasVnrType
* MasVnrArea
* BsmtQual
* BsmtCond
* BsmtExposure
* BsmtFinType1
* BsmtFinType2
* Electrical
* FireplaceQu
* GarageType
* GarageYrBlt
* GarageFinish
* GarageQual
* GarageCond
* PoolQC
* Fence
* MiscFeature

Test
------------

* MSZoning
* LotFrontage
* Alley
* Utilities
* Exterior1st
* Exterior2nd
* MasVnrType
* MasVnrArea
* BsmtQual
* BsmtCond
* BsmtExposure
* BsmtFinType1
* BsmtFinSF1
* BsmtFinType2
* BsmtFinSF2
* BsmtUnfSF
* TotalBsmtSF
* BsmtFullBath
* BsmtHalfBath
* KitchenQual
* Functional
* FireplaceQu
* GarageType
* GarageYrBlt
* GarageFinish
* GarageCars
* GarageArea
* GarageQual
* GarageCond
* PoolQC
* Fence
* MiscFeature
* SaleType

Reformatting
============

* MSSubClass: 15 categorical classes -- use label encoder and scale

* MSZoning: 8 categorical classes -- use label encoder and scale -- impute with mode

* LotFrontage: real value distance between property and street -- lots of nans -- impute with 0 or mean -- scale

* LotArea: int valued square footage -- scale

* Street: 2 categorical classes -- label encode and scale

* Alley: 3 categorical classes -- label encode and scale -- impute nan with NA (no alley access)

* LotShape: 4 categorical classes -- label encode and scale

* LandContour: 4 categorical classes -- label encode and scale

* Utilities: 4 categorical classes -- label encode and scale -- impute nan with mode

* LotConfig: 5 categorical classes -- label encode and scale

* LandSlope: 3 categorical classes -- label encode and scale

* Neighborhood: 25 categorical classes -- label encode and scale

* Condition1: 9 categorical classes -- label encode and scale

* Condition2: 9 categorical classes -- label encode and scale

* BldgType: 5 categorical classes -- label encode and scale

* HouseStyle: 8 categorical classes -- label encode and scale

* OverallQual: 10 categorical classes

* OverallCond: 10 categorical classes

* YearBuilt: int valued dates -- label encode and scale

* YearRemodAdd: int valued dates -- label encode and scale

* RoofStyle: 6 categorical classes -- label encode and scale

* RoofMatl: 8 categorical classes -- label encode and scale

* Exterior1st: 17 categorical classes -- label encode and scale -- impute with mode

* Exterior2nd: 17 categorical classes -- label encode and scale -- impute with mode

* MasVnrType: 5 categorical classes -- label encode and scale -- impute with mode

* MasVnrArea: real valued -- scale -- impute with mean

* ExterQual: 5 categorical classes -- label encode and scale

* ExterCond: 5 categorical classes -- label encode and scale

* Foundation: 6 categorical classes -- label encode and scale

* BsmtQual: 5 categorical classes -- label encode and scale -- impute with NA

* BsmtCond: 5 categorical classes -- label encode and scale -- impute with NA

* BsmtExposure: 5 categorical classes -- label encode and scale -- impute with NA

* BsmtFinType1: 7 categorical classes -- label encode and scale -- impute with NA

* BsmtFinSF1: real valued -- scale -- impute with mean

* BsmtFinType2: 7 categorical classes -- label encode and scale -- impute with NA

* BsmtFinSF2: real valued -- scale -- impute with mean

* BsmtUnfSF: real valued -- scale -- inpute with mean

* TotalBsmtSF: real valued -- scale -- impute with mean

* Heating: 6 categorical classes -- label encode and scale

* HeatingQC: 5 categorical classes -- label encode and scale

* CentralAir: 2 categorical classes -- label encode and scale

* Electrical: 5 categorical classes -- label encode and scale -- impute with mode

* 1stFlrSF: int valued -- scale

* 2ndFlrSF: int valued -- scale

* LowQualFinSF: int valued -- scale

* GrLivArea: int valued -- scale

* BsmtFullBath: int valued -- scale -- impute with mode

* BsmtHalfBath: int valued -- scale -- impute with mode

* FullBath: int valued -- scale

* HalfBath: int valued -- scale

* BedroomAbvGr: int value d-- scale

* KitchenAbvGr: int valued -- scale

* KitchenQual: 5 categorical classes -- label encode and scale -- impute with mode

* TotRmsAbvGr: int valued -- scale

* Functional: 8 categorical classes -- label encode and scale -- impute with mode

* Fireplaces: int valued -- scale

* FireplaceQu: 6 categorical classes -- label encode and scale -- impute with NA

* GarageType: 7 categorical classes -- label encode and scale -- impute with NA

* GarageYrBlt: int valued -- scale -- impute with 0

* GarageFinish: 4 categorical classes -- label encode and scale -- impute with NA

* GarageCars: int valued -- scale -- impute with mode

* GarageArea: real valued -- scale -- impute with mean

* GarageQual: 6 categorical classes -- label encode and scale -- impute with NA

* GarageCond: 6 categorical classes -- label encode and scale -- impute with NA

* PavedDrive: 3 categorical classes -- label encode and scale

* WoodDeckSF: int valued -- scale

* OpenPorchSF: int valued -- scale

* EnclosedPorch: int valued -- scale

* 3SsnPorch: int valued -- scale

* ScreenPorch: int valued -- scale

* PoolArea: int valued -- scale

* PoolQC: 5 categorical classes -- label encode and scale -- impute with NA

* Fence: 5 categorical classes -- label encode and scale -- impute with NA

* MiscFeature: 6 categorical classes -- label encode and scale -- impute with NA

* MiscVal: int valued -- scale

* MoSold: 12 categorical classes -- label encode and scale

* YrSold: int valued -- scale

* SaleType: 10 categorical classes -- label encode and scale -- impute with mode

* SaleCondition: 6 categorical classes -- label encode and scale
