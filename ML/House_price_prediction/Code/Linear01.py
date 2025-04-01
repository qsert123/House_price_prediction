import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
import numpy as np
#Load the data
df = pd.read_csv('D:\pythonproject\Project01\ML\House_price_prediction\data\housing.csv')

#check for missing value
# print(df.isnull().sum())

#handle the missing value
df = df.assign(total_bedrooms=lambda x: x['total_bedrooms'].fillna(x['total_bedrooms'].median()))

#create a new feature
df = df.assign(
    rooms_per_household = lambda x: x['total_rooms']/x['households'],
    bedrooms_per_room = lambda x: x['total_bedrooms']/x['total_rooms'],
    population_per_household = lambda x:x['population']/x['households']
)

numerical_feature = ['longitude', 'latitude', 'housing_median_age',
                     'total_rooms', 'total_bedrooms', 'population',
                     'households', 'median_income', 'rooms_per_household',
                     'bedrooms_per_room', 'population_per_household']
categorical_features = ['ocean_proximity']
target = 'median_house_value'

x = df[numerical_feature]
y = df[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=32)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
new_data = pd.DataFrame({
    'longitude': [-122.25],
    'latitude': [37.85],
    'housing_median_age': [2],
    'total_rooms': [1274],
    'total_bedrooms': [235],
    'population': [558],
    'households': [219],
    'median_income': [5.6431],
    'rooms_per_household': [1274/219],
    'bedrooms_per_room': [235/1274],
    'population_per_household': [558/219]
})
predicted_price=model.predict(new_data)
print(model.predict(new_data))
print(f"The house price will be: ${predicted_price[0]:,.1f}")

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)


test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)


print("Training set metrics")
print(f"MAE: ${train_mae:,.2f}")
print(f"MSE: ${train_mse:,.2f}")
print(f"RMSE: ${train_rmse:,.2f}")
print(f"R2_Score: ${train_r2:,.2f}")

print("Test set Metrics")
print(f"MAE: ${test_mae:,.2f}")
print(f"MSE: ${test_mse:,.2f}")
print(f"RMSE: ${test_rmse:,.2f}")
print(f"R2_score: ${test_r2:,.2f}")
