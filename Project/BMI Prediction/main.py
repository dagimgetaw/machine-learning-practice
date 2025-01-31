import pandas as pd
from sklearn.preprocessing import LabelEncoder

bmi_dataset = pd.read_csv("bmi.csv")
bmi_dataset.drop_duplicates(inplace=True)

label_encoder = LabelEncoder()
bmi_dataset['Gender'] = label_encoder.fit_transform(bmi_dataset['Gender'])

print(bmi_dataset.head(10))
