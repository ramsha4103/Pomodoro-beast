                                          #### DATA IMPORTED AS A LIST OF LISTS ####
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

#Plotting the data
import matplotlib.pyplot as plt

#Breaking down the lists for plotting
DataX = [x[0] for x in Total]
DataY = [x[1] for x in Total]

print("List 1:", DataX)
print("List 2:", DataY)

plt.xlabel("Expected Number of Pomodoros")
plt.ylabel("Actual Number of Pomodoros")
plt.scatter(DataX, DataY,alpha=0.5)
plt.show

                                                ####  DESCRIPTIVE LINEAR REGRESSION ####
# Create a linear regression model
model = LinearRegression()

# Fit the model to your data
model.fit(DataX, DataY)

# Make predictions
y_pred = model.predict(DataX)

# Visualize the data and regression line
plt.scatter(DataX, DataY, alpha=0.5, label="Actual Data")
plt.plot(DataX, y_pred, color='red', label="Regression Line")
plt.xlabel("Expected Number of Pomodoros")
plt.ylabel("Actual Number of Pomodoros")
plt.legend()
plt.show()

                                                 #### LINEAR REGRESSION WITH A 80/20 SPLIT ####
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeRegressor

# Reshape the data to be 2D (required by scikit-learn)
DataX = np.array(DataX).reshape(-1, 1)
DataY = np.array(DataY)

# Perform an 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(DataX, DataY, test_size=0.8)

# Create a polynomial regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train_poly, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test_poly)

mse = mean_squared_error(y_test, y_pred)
# Calculate root mean squared error
rmse = np.sqrt(mse)


# Calculate mean absolute error
mae = mean_absolute_error(y_test, y_pred)

# Visualize the data and regression line
plt.scatter(DataX, DataY, alpha=0.5, label="Actual Data")
plt.scatter(X_test, y_pred, color='red', label="Predicted Data")
plt.xlabel("Expected Number of Pomodoros")
plt.ylabel("Actual Number of Pomodoros")
plt.title('Linear Regression with 80/20 split')
plt.legend()
plt.show()

# Print the regression coefficients, RMSE, and MAE
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae

                                             #### LINEAR REGRESSION WITH LOOCV ####
# Initialize lists to store the predicted and actual values
y_pred_loocv = []
y_actual_loocv = []

loo = LeaveOneOut()
for train_index, test_index in loo.split(DataX):
    X_train, X_test = DataX[train_index], DataX[test_index]
    y_train, y_test = DataY[train_index], DataY[test_index]

    poly_features = PolynomialFeatures(degree=2)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)

    y_pred_loocv.append(y_pred[0])
    y_actual_loocv.append(y_test[0])

# Convert lists to NumPy arrays
y_pred_loocv = np.array(y_pred_loocv)
y_actual_loocv = np.array(y_actual_loocv)

# Plotting the LOOCV predictions and actual values
plt.scatter(DataX, DataY, color='blue', label="Actual Data")
plt.scatter(DataX, y_pred_loocv, color='red', label="Predicted Data")
plt.xlabel("Expected Number of Pomodoros")
plt.ylabel("Actual/Predicted Number of Pomodoros")
plt.title('LOOCV for Linear Regression')
plt.legend()
plt.show()

# Calculate mean squared error for LOOCV
mse_loocv = mean_squared_error(y_actual_loocv, y_pred_loocv)
rmse_loocv = np.sqrt(mse_loocv)
mae_loocv = mean_absolute_error(y_actual_loocv, y_pred_loocv)

# Print mean squared error, root mean squared error, and mean absolute error
print(f"Root Mean Squared Error with LOOCV: {rmse_loocv}")
print(f"Mean Absolute Error with LOOCV: {mae_loocv}")

                                          #### LOGISTIC REGRESSION TREE ####
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

                                 #### LOGISTIC REGRESSION TREE WITH LOOCV ####
# Perform Leave-One-Out Cross-Validation
loo = LeaveOneOut()
y_true_cv, y_pred_cv = [], []

for train_index, test_index in loo.split(DataX):
    X_train_cv, X_test_cv = DataX[train_index], DataX[test_index]
    y_train_cv, y_test_cv = DataY[train_index], DataY[test_index]

    # Fit the regression tree on the training data
    reg_tree_cv = DecisionTreeRegressor(random_state=0)
    reg_tree_cv.fit(X_train_cv, y_train_cv)

    # Make predictions on the test data
    y_pred_cv.extend(reg_tree_cv.predict(X_test_cv))
    y_true_cv.extend(y_test_cv)

# Compute evaluation metrics for LOOCV
mae_cv = mean_absolute_error(y_true_cv, y_pred_cv)
mse_cv = mean_squared_error(y_true_cv, y_pred_cv)
rmse_cv = np.sqrt(mse_cv)

# Print the evaluation metrics for LOOCV
print(f"LOOCV - Mean Absolute Error: {mae_cv:.2f}")
print(f"LOOCV - Root Mean Squared Error: {rmse_cv:.2f}")

import matplotlib.pyplot as plt

# Create a scatter plot of true values vs predicted values for LOOCV
plt.scatter(DataX, DataY, color='red', label='Predicted Values')
plt.scatter(DataX, y_pred_cv, color='blue', label='Predicted Values')
plt.title('LOOCV for Regression Tree')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()

                                 #### DESCRIPTIVE LINEAR REGRESSION AFTER SPLITTING THE DATA ####

DataZ = [1,0,0,0,0,0,1,1,0,0,0,0,1,0,0]

# Create a colormap
colors = ['green' if z == 1 else 'orange' for z in DataZ]

# Plotting the data
plt.scatter(DataX, DataY, c=colors, alpha=0.5)
plt.xlabel("Expected Number of Pomodoros")
plt.ylabel("Actual Number of Pomodoros")
plt.title('Expected and predicted pomodoros, green for exciting tasks and yellow for chores')

plt.show()

# Filter data based on DataZ values
green_indices = [i for i, z in enumerate(DataZ) if z == 1]
yellow_indices = [i for i, z in enumerate(DataZ) if z == 0]

# Fit separate linear regression models for red and blue points
model_green = LinearRegression()
model_yellow = LinearRegression()

model_green.fit(DataX[green_indices], DataY[green_indices])
model_yellow.fit(DataX[yellow_indices], DataY[yellow_indices])

# Make predictions
y_pred_green = model_green.predict(DataX)
y_pred_yellow = model_yellow.predict(DataX)

# Calculate mean squared error
mse_green = mean_squared_error(DataY[green_indices], y_pred_green[green_indices])
mse_yellow = mean_squared_error(DataY[yellow_indices], y_pred_yellow[yellow_indices])

# Visualize the data and regression lines
plt.scatter(DataX[green_indices], DataY[green_indices], color='green', alpha=0.5, label="green Data")
plt.scatter(DataX[yellow_indices], DataY[yellow_indices], color='yellow', alpha=0.5, label="yellow Data")
plt.plot(DataX, y_pred_green, color='green', label="green Regression Line")
plt.plot(DataX, y_pred_yellow, color='yellow', label="yellow Regression Line")
plt.xlabel("Expected Number of Pomodoros")
plt.ylabel("Actual Number of Pomodoros")
plt.legend()
plt.show()

                            #### LINEAR REGRESSION ON DATA AFTER SPLITTING WITH A 80/20 SPLIT ####
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# THE COLORS ARE CHANGED WITHIN THE VARIABLES SO I COULD TELL THE DIFFERENCE BETWEEN THEM :D

# Your data
DataX = np.array([x[0] for x in Total]).reshape(-1, 1)
DataY = np.array([x[1] for x in Total])
DataZ = np.array([1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0])

# Red Data
red_X = DataX[DataZ == 1]
red_Y = DataY[DataZ == 1]

# Blue Data
blue_X = DataX[DataZ == 0]
blue_Y = DataY[DataZ == 0]

# Polynomial degree
degree = 2

# Create polynomial features for red data
poly_features_red = PolynomialFeatures(degree=degree)
red_X_poly = poly_features_red.fit_transform(red_X)
red_model = LinearRegression()
red_model.fit(red_X_poly, red_Y)
red_Y_pred = red_model.predict(red_X_poly)

# Create polynomial features for blue data
poly_features_blue = PolynomialFeatures(degree=degree)
blue_X_poly = poly_features_blue.fit_transform(blue_X)
blue_model = LinearRegression()
blue_model.fit(blue_X_poly, blue_Y)
blue_Y_pred = blue_model.predict(blue_X_poly)

# Visualize the data and predicted values for red data
plt.scatter(red_X, red_Y, color='green', label="Green Data")
plt.scatter(red_X, red_Y_pred, color='red', label="Predicted Data")
plt.xlabel("Expected Number of Pomodoros")
plt.ylabel("Actual/Predicted Number of Pomodoros")
plt.legend()
plt.show()

# Visualize the data and predicted values for blue data
plt.scatter(blue_X, blue_Y, color='yellow', label="Yellow Data")
plt.scatter(blue_X, blue_Y_pred, color='red', label="Predicted Data")
plt.xlabel("Expected Number of Pomodoros")
plt.ylabel("Actual/Predicted Number of Pomodoros")
plt.legend()
plt.show()


# Calculate RMSE and MAE for red data
rmse_red = np.sqrt(mean_squared_error(red_Y, red_Y_pred))
mae_red = np.mean(np.abs(red_Y - red_Y_pred))

# Calculate RMSE and MAE for blue data
rmse_blue = np.sqrt(mean_squared_error(blue_Y, blue_Y_pred))
mae_blue = np.mean(np.abs(blue_Y - blue_Y_pred))

# Print the regression coefficients and mean squared errors for red data
print("Green Data:")
print("Intercept:", red_model.intercept_)
print("Coefficient:", red_model.coef_)
print("Mean Squared Error:", mean_squared_error(red_Y, red_Y_pred))
print("Root Mean Squared Error:", rmse_red)
print("Mean Absolute Error:", mae_red)

# Print the regression coefficients and mean squared errors for blue data
print("\nYellow Data:")
print("Intercept:", blue_model.intercept_)
print("Coefficient:", blue_model.coef_)
print("Mean Squared Error:", mean_squared_error(blue_Y, blue_Y_pred))
print("Root Mean Squared Error:", rmse_blue)
print("Mean Absolute Error:", mae_blue)



# Calculate average RMSE and MAE
avg_rmse = (rmse_red + rmse_blue) / 2
avg_mae = (mae_red + mae_blue) / 2

# Print the average errors
print("\nAverage RMSE:", avg_rmse)
print("Average MAE:", avg_mae)

                                #### LOOCV ON LINEAR REGRESSION AFTER SPLITTING ####
# Your data
DataX = np.array([x[0] for x in Total]).reshape(-1, 1)
DataY = np.array([x[1] for x in Total])
DataZ = np.array([1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0])

# Initialize lists to store the predicted and actual values
red_pred_loocv = []
red_actual_loocv = []
blue_pred_loocv = []
blue_actual_loocv = []

loo = LeaveOneOut()
for train_index, test_index in loo.split(DataX):
    X_train, X_test = DataX[train_index], DataX[test_index]
    y_train, y_test = DataY[train_index], DataY[test_index]
    z_test = DataZ[test_index]

    poly_features = PolynomialFeatures(degree=2)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)

    if z_test == 1:
        red_pred_loocv.append(y_pred[0])
        red_actual_loocv.append(y_test[0])
    else:
        blue_pred_loocv.append(y_pred[0])
        blue_actual_loocv.append(y_test[0])

# Convert lists to NumPy arrays
red_pred_loocv = np.array(red_pred_loocv).reshape(-1, 1)
red_actual_loocv = np.array(red_actual_loocv).reshape(-1, 1)
blue_pred_loocv = np.array(blue_pred_loocv).reshape(-1, 1)
blue_actual_loocv = np.array(blue_actual_loocv).reshape(-1, 1)

# Plotting the red LOOCV predictions and actual values
plt.scatter(DataX[DataZ == 1], red_actual_loocv, color='blue', label="Red Actual Data")
plt.scatter(DataX[DataZ == 1], red_pred_loocv, color='red', label="Red Predicted Data")
plt.xlabel("Expected Number of Pomodoros")
plt.ylabel("Actual/Predicted Number of Pomodoros")
plt.legend()
plt.show()

# Plotting the blue LOOCV predictions and actual values
plt.scatter(DataX[DataZ == 0], blue_actual_loocv, color='blue', label="Blue Actual Data")
plt.scatter(DataX[DataZ == 0], blue_pred_loocv, color='red', label="Blue Predicted Data")
plt.xlabel("Expected Number of Pomodoros")
plt.ylabel("Actual/Predicted Number of Pomodoros")
plt.legend()
plt.show()

# Calculate mean squared error for LOOCV for red and blue data
red_mse_loocv = mean_squared_error(red_actual_loocv, red_pred_loocv)
blue_mse_loocv = mean_squared_error(blue_actual_loocv, blue_pred_loocv)

# Calculate RMSE and MAE for red data
red_rmse_loocv = np.sqrt(red_mse_loocv)
red_mae_loocv = np.mean(np.abs(red_actual_loocv - red_pred_loocv))

# Calculate RMSE and MAE for blue data
blue_rmse_loocv = np.sqrt(blue_mse_loocv)
blue_mae_loocv = np.mean(np.abs(blue_actual_loocv - blue_pred_loocv))


# Calculate average RMSE and MAE
avg_rmse = (red_rmse_loocv + blue_rmse_loocv) / 2
avg_mae = (red_mae_loocv + blue_mae_loocv) / 2

# Print the average errors
print("\nAverage RMSE:", avg_rmse)
print("Average MAE:", avg_mae)

                                   #### REGRESSION TREES AFTER THE SPLIT ####

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

z = np.array([1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0])
DataX = np.array([x[0] for x in Total]).reshape(-1, 1)
DataY = np.array([x[1] for x in Total])

# Create and fit the first regression tree for z == 0
reg_tree_0 = DecisionTreeRegressor(random_state=0)
reg_tree_0.fit(DataX[z == 0], DataY[z == 0])

# Create and fit the second regression tree for z == 1
reg_tree_1 = DecisionTreeRegressor(random_state=0)
reg_tree_1.fit(DataX[z == 1], DataY[z == 1])

# Visualize the first regression tree for z == 0
X_grid_0 = np.arange(min(DataX[z == 0]), max(DataX[z == 0]), 0.01).reshape(-1, 1)
plt.scatter(DataX[z == 0], DataY[z == 0], color='red', label='Data Points for Z == 0')
plt.plot(X_grid_0, reg_tree_0.predict(X_grid_0), color='blue', label='Regression Tree Prediction for Z == 0')
plt.title('Regression Tree for Z == 0')
plt.xlabel('Data X for Z == 0')
plt.ylabel('Data Y for Z == 0')
plt.legend()
plt.show()

# Visualize the second regression tree for z == 1
X_grid_1 = np.arange(min(DataX[z == 1]), max(DataX[z == 1]), 0.01).reshape(-1, 1)
plt.scatter(DataX[z == 1], DataY[z == 1], color='green', label='Data Points for Z == 1')
plt.plot(X_grid_1, reg_tree_1.predict(X_grid_1), color='orange', label='Regression Tree Prediction for Z == 1')
plt.title('Regression Tree for Z == 1')
plt.xlabel('Data X for Z == 1')
plt.ylabel('Data Y for Z == 1')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Make predictions for z == 0
y_pred_0 = reg_tree_0.predict(DataX[z == 0])
rmse_0 = np.sqrt(mean_squared_error(DataY[z == 0], y_pred_0))
mae_0 = mean_absolute_error(DataY[z == 0], y_pred_0)
print(f"For Z == 0 - RMSE: {rmse_0:.2f}")
print(f"For Z == 0 - MAE: {mae_0:.2f}")

# Make predictions for z == 1
y_pred_1 = reg_tree_1.predict(DataX[z == 1])
rmse_1 = np.sqrt(mean_squared_error(DataY[z == 1], y_pred_1))
mae_1 = mean_absolute_error(DataY[z == 1], y_pred_1)
print(f"For Z == 1 - RMSE: {rmse_1:.2f}")
print(f"For Z == 1 - MAE: {mae_1:.2f}")


# Calculate average RMSE and MAE
avg_rmse = (rmse_0 + rmse_1) / 2
avg_mae = (mae_0 + mae_1) / 2

# Print the average errors
print("\nAverage RMSE:", avg_rmse)
print("Average MAE:", avg_mae)


                                        #### LOOCV ON REGRESSION TREES AFTER THE SPLIT ####
import matplotlib.pyplot as plt

# LOOCV for z == 0
loo = LeaveOneOut()
y_true, y_pred = [], []

for train_index, test_index in loo.split(DataX[z == 0]):
    X_train, X_test = DataX[z == 0][train_index], DataX[z == 0][test_index]
    y_train, y_test = DataY[z == 0][train_index], DataY[z == 0][test_index]

    reg_tree_0_cv = DecisionTreeRegressor(random_state=0)
    reg_tree_0_cv.fit(X_train, y_train)

    y_pred.extend(reg_tree_0_cv.predict(X_test))
    y_true.extend(y_test)

# Visualize LOOCV for z == 0
plt.scatter(y_true, y_pred, color='blue', label='Predicted vs True Values for Z == 0')
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', label='Ideal Prediction Line')
plt.title('LOOCV for Regression Tree - Z == 0')
plt.xlabel('True Values for Z == 0')
plt.ylabel('Predicted Values for Z == 0')
plt.legend()
plt.show()

# Compute evaluation metrics for LOOCV - z == 0
rmse_0_cv = np.sqrt(mean_squared_error(y_true, y_pred))
mae_0_cv = mean_absolute_error(y_true, y_pred)
print(f"LOOCV - Z == 0 - RMSE: {rmse_0_cv:.2f}")
print(f"LOOCV - Z == 0 - MAE: {mae_0_cv:.2f}")

# LOOCV for z == 1
y_true, y_pred = [], []

for train_index, test_index in loo.split(DataX[z == 1]):
    X_train, X_test = DataX[z == 1][train_index], DataX[z == 1][test_index]
    y_train, y_test = DataY[z == 1][train_index], DataY[z == 1][test_index]

    reg_tree_1_cv = DecisionTreeRegressor(random_state=0)
    reg_tree_1_cv.fit(X_train, y_train)

    y_pred.extend(reg_tree_1_cv.predict(X_test))
    y_true.extend(y_test)

# Visualize LOOCV for z == 1
plt.scatter(y_true, y_pred, color='green', label='Predicted vs True Values for Z == 1')
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='orange', linestyle='--', label='Ideal Prediction Line')
plt.title('LOOCV for Regression Tree - Z == 1')
plt.xlabel('True Values for Z == 1')
plt.ylabel('Predicted Values for Z == 1')
plt.legend()
plt.show()

# Compute evaluation metrics for LOOCV - z == 1
rmse_1_cv = np.sqrt(mean_squared_error(y_true, y_pred))
mae_1_cv = mean_absolute_error(y_true, y_pred)
print(f"LOOCV - Z == 1 - RMSE: {rmse_1_cv:.2f}")
print(f"LOOCV - Z == 1 - MAE: {mae_1_cv:.2f}")

# Calculate average RMSE and MAE
avg_rmse = (rmse_0_cv + rmse_1_cv) / 2
avg_mae = (mae_0_cv + mae_1_cv) / 2

# Print the average errors
print("\nAverage RMSE:", avg_rmse)
print("Average MAE:", avg_mae)
