import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeRegressor
Total =([[3, 5],
       [5, 4],
       [1, 2],
       [2, 3],
       [1, 1],
       [1, 2],
       [2, 6],
       [1, 3],
       [2, 3],
       [1, 1],
       [1, 2],
       [1, 1],
       [3, 10],
       [2, 3],
       [1, 1]])

# Reshape the data to be 2D (required by scikit-learn)
DataX = np.array(DataX).reshape(-1, 1)
DataY = np.array(DataY)

# Perform an 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(DataX, DataY, test_size=0.8)

# Create and fit the regression tree on the training data
reg_tree_split = DecisionTreeRegressor(random_state=0)
reg_tree_split.fit(X_train, y_train)

# Make predictions on the test data
y_pred_split = reg_tree_split.predict(X_test)

# Compute evaluation metrics for the 80/20 split
mae_split = mean_absolute_error(y_test, y_pred_split)
mse_split = mean_squared_error(y_test, y_pred_split)
rmse_split = np.sqrt(mse_split)


# Print the evaluation metrics for the 80/20 split
print("Mean Absolute Error", mae_split)
print("Root Mean Squared Error:", rmse_split)


# Visualize the regression tree with the 80/20 split data
#X_grid = np.arange(min(DataX), max(DataX), 0.01).reshape(-1, 1)
plt.scatter(DataX, DataY, color='blue', label='Actual Data Points')
plt.scatter(X_test, y_pred_split, color='red', label='Predicted Data Points')
plt.plot(X_grid, reg_tree_split.predict(X_grid), color='blue', label='Regression Tree Prediction')
plt.title('Logistic Tree with 80/20 Split')
plt.xlabel('Estimated Pomodoros')
plt.ylabel('Actual Pomodoros')
plt.legend()
plt.show()
