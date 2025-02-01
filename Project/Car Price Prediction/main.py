import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

car_dataset = pd.read_csv('car data.csv')
car_dataset.drop_duplicates(inplace=True)

col = car_dataset.columns.to_list()
print(col)

print(car_dataset.Fuel_Type.value_counts())
print(car_dataset.Seller_Type.value_counts())
print(car_dataset.Transmission.value_counts())

label_encoder = LabelEncoder()
car_dataset['Fuel_Type'] = label_encoder.fit_transform(car_dataset['Fuel_Type'])
car_dataset['Seller_Type'] = label_encoder.fit_transform(car_dataset['Seller_Type'])
car_dataset['Transmission'] = label_encoder.fit_transform(car_dataset['Transmission'])
car_dataset.head()

x = car_dataset.drop(columns=['Car_Name', 'Selling_Price'], axis=1)
y = car_dataset['Selling_Price']
print(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)

linear_model = LinearRegression()
linear_model.fit(x_train, y_train)

linear_train_prediction = linear_model.predict(x_train)
linear_test_prediction = linear_model.predict(x_test)

linear_train_r2 = r2_score(y_train, linear_train_prediction)
linear_test_r2 = r2_score(y_test, linear_test_prediction)

linear_train_mse = mean_squared_error(y_train, linear_train_prediction)
linear_test_mse = mean_squared_error(y_test, linear_test_prediction)


print(f"Linear train prediction = r2 error {linear_train_r2:.4f}, mse {linear_train_mse:.4f}")
print(f"Linear test prediction = r2 error {linear_test_r2:.4f}, mse {linear_test_mse:.4f}")

plt.figure(figsize=(6, 4))
plt.scatter(y_train, linear_train_prediction, color='blue', alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear Regression - Actual vs Predicted Prices")
plt.show()

plt.figure(figsize=(6, 4))
plt.scatter(y_test, linear_test_prediction, color='green', alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear Regression - Test Data (Actual vs Predicted)")
plt.show()

lasso_model = Lasso()
lasso_model.fit(x_train, y_train)

lasso_train_prediction = lasso_model.predict(x_train)
lasso_test_prediction = lasso_model.predict(x_test)

lasso_train_r2 = r2_score(y_train, lasso_train_prediction)
lasso_test_r2 = r2_score(y_test, lasso_test_prediction)

lasso_train_mse = mean_squared_error(y_train, lasso_train_prediction)
lasso_test_mse = mean_squared_error(y_test, lasso_test_prediction)

print(f"Lasso train prediction = r2 error {lasso_train_r2:.4f}, mse {lasso_train_mse:.4f}")
print(f"Lasso train prediction = r2 error {lasso_test_r2:.4f}, mse {lasso_test_mse:.4f}")

plt.figure(figsize=(6, 4))
plt.scatter(y_train, lasso_train_prediction, color='red', alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Lasso Regression - Actual vs Predicted Prices")
plt.show()

plt.figure(figsize=(6, 4))
plt.scatter(y_test, lasso_test_prediction, color='purple', alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Lasso Regression - Test Data (Actual vs Predicted)")
plt.show()
