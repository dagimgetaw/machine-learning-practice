import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

bmi_dataset = pd.read_csv("bmi.csv")
bmi_dataset.drop_duplicates(inplace=True)

label_encoder = LabelEncoder()
bmi_dataset['Gender'] = label_encoder.fit_transform(bmi_dataset['Gender'])

x = bmi_dataset.drop(columns=['Index'])
y = bmi_dataset['Index']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

prediction = model.predict(x_test)

print(bmi_dataset.head(10))
