# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
print ('---------------------------------------------')
# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
print ('---------------------------------------------')
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#avoiding the dummy variable trap
X= X[:,1:]


# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#backward elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X , axis = 1 )

X_opt = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS( endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = np.array(X[:, [0, 1, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS( endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = np.array(X[:, [0, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS( endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = np.array(X[:, [0, 3, 5]], dtype=float)
regressor_OLS = sm.OLS( endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = np.array(X[:, [0, 3]], dtype=float)
regressor_OLS = sm.OLS( endog = y, exog = X_opt).fit()
regressor_OLS.summary()






















