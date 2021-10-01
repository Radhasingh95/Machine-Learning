import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import mean
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()

# ['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename']

# print(diabetes.data)
# print(diabetes.keys())

diabetes_X = np.array(([1],[2],[3]))

diabetes_X_train = diabetes_X
diabetes_X_test = diabetes_X

diabetes_y_train = np.array([3,2,4])
diabetes_y_test = np.array([3,2,4])

model = linear_model.LinearRegression()

model.fit(diabetes_X_train, diabetes_y_train)

diabetes_y_predicted =  model.predict(diabetes_X_test)


print("Mean squared error is: ", mean_squared_error(diabetes_y_test,diabetes_y_predicted))

print("Weights: ", model.coef_)
print("Intercept",model.intercept_)

# plt.scatter(diabetes_X_test, diabetes_y_test)
# plt.plot(diabetes_X_test, diabetes_y_predicted)
# # plt.show()
# Mean squared error is:  3035.0601152912695    
# Weights:  [941.43097333]
# Intercept 153.39713623331698