# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load historical stock data
# Replace 'stock_data.csv' with your dataset
data = pd.read_csv('stock_data.csv')

# Data preprocessing
# Assuming your dataset contains 'Date', 'Open', 'High', 'Low', 'Close' columns
# You may need additional preprocessing based on your dataset

# Feature selection
# For simplicity, let's consider 'Open', 'High', 'Low' as features and 'Close' as target
X = data[['Open', 'High', 'Low']]
y = data['Close']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the ensemble regressors
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
gradient_boosting = GradientBoostingRegressor(n_estimators=100, random_state=42)
adaboost = AdaBoostRegressor(n_estimators=100, random_state=42)
extra_trees = ExtraTreesRegressor(n_estimators=100, random_state=42)
bagging = BaggingRegressor(base_estimator=LinearRegression(), n_estimators=100, random_state=42)
voting = VotingRegressor([('rf', random_forest), ('gb', gradient_boosting), ('ab', adaboost), ('et', extra_trees)])

# Training the models
random_forest.fit(X_train, y_train)
gradient_boosting.fit(X_train, y_train)
adaboost.fit(X_train, y_train)
extra_trees.fit(X_train, y_train)
bagging.fit(X_train, y_train)
voting.fit(X_train, y_train)

# Making predictions
rf_predictions = random_forest.predict(X_test)
gb_predictions = gradient_boosting.predict(X_test)
ab_predictions = adaboost.predict(X_test)
et_predictions = extra_trees.predict(X_test)
bagging_predictions = bagging.predict(X_test)
voting_predictions = voting.predict(X_test)

# Evaluating the models
rf_mse = mean_squared_error(y_test, rf_predictions)
gb_mse = mean_squared_error(y_test, gb_predictions)
ab_mse = mean_squared_error(y_test, ab_predictions)
et_mse = mean_squared_error(y_test, et_predictions)
bagging_mse = mean_squared_error(y_test, bagging_predictions)
voting_mse = mean_squared_error(y_test, voting_predictions)

print("Random Forest Mean Squared Error:", rf_mse)
print("Gradient Boosting Mean Squared Error:", gb_mse)
print("AdaBoost Mean Squared Error:", ab_mse)
print("Extra Trees Mean Squared Error:", et_mse)
print("Bagging Mean Squared Error:", bagging_mse)
print("Voting Mean Squared Error:", voting_mse)

# Visualizing the predictions
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(rf_predictions, label='Random Forest Predicted')
plt.plot(gb_predictions, label='Gradient Boosting Predicted')
plt.plot(ab_predictions, label='AdaBoost Predicted')
plt.plot(et_predictions, label='Extra Trees Predicted')
plt.plot(bagging_predictions, label='Bagging Predicted')
plt.plot(voting_predictions, label='Voting Predicted')
plt.xlabel('Index')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction with Ensemble Learning')
plt.legend()
plt.show()
