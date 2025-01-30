import pandas as pd

bmi_dataset = pd.read_csv("bmi.csv")
bmi_dataset.drop_duplicates(inplace=True)

