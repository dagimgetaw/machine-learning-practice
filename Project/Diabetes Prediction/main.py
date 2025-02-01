import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

db_dataset = pd.read_csv('diabetes.csv')
print(db_dataset.columns.to_list())

x = db_dataset.drop(columns='Outcome', axis=1)
y = db_dataset['Outcome']
print(x, y)

scaler = StandardScaler()
scaler.fit(x)

standardize_data = scaler.transform(x)
print(standardize_data)

x = standardize_data
y = db_dataset['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
print(x.shape, x_train.shape, x_test.shape)

model = SVC(kernel='linear')
model.fit(x_train, y_train)

train_prediction = model.predict(x_train)
train_accuracy = accuracy_score(y_train, train_prediction)
train_mse = mean_squared_error(y_train, train_prediction)
print(f"The train model accuracy is {train_accuracy}, mse {train_mse}")

test_prediction = model.predict(x_test)
test_accuracy = accuracy_score(test_prediction, y_test)
test_mse = mean_squared_error(test_prediction, y_test)
print(f"The train model accuracy is {test_accuracy}, mse {test_mse}")

input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)
input_data_df = pd.DataFrame([input_data], columns=db_dataset.columns[:-1])
std_data = scaler.transform(input_data_df)

prediction = model.predict(std_data)
if prediction[0] == 0:
    print("The person doesn't have Diabetes")
else:
    print("The peron have Diabetes")
