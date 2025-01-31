import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


bmi_dataset = pd.read_csv("bmi.csv")
bmi_dataset.drop_duplicates(inplace=True)

plt.figure(figsize=(4, 4))
sns.countplot(x='Index', data=bmi_dataset)
plt.title('Distribution of Obesity Levels')
plt.xticks(rotation=90)
plt.show()

label_encoder = LabelEncoder()
bmi_dataset['Gender'] = label_encoder.fit_transform(bmi_dataset['Gender'])

scale = MinMaxScaler()
bmi_dataset[['Height', 'Weight']] = scale.fit_transform(bmi_dataset[['Height', 'Weight']])

x = bmi_dataset.drop(columns=['Index'])
y = bmi_dataset['Index']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

prediction = model.predict(x_test)

accuracy = accuracy_score(y_test, prediction)
print(f"accuracy {accuracy}")

conf_matrix = confusion_matrix(y_test, prediction)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# Define new sample
new_sample = pd.DataFrame([['Male', 174, 96]], columns=['Gender', 'Height', 'Weight'])

new_sample['Gender'] = label_encoder.transform(new_sample['Gender'])
new_sample[['Height', 'Weight']] = scale.transform(new_sample[['Height', 'Weight']])
predicted_bmi = model.predict(new_sample)

print(f"Predicted BMI Index: {predicted_bmi[0]}")


