import pandas as pd  # use for data manipulation and analysis

# load the csv file from the given URL and store as Dataframe
chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")
# select specific colum we want and store as Dataframe
training_df = chicago_taxi_dataset[['TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE']]
print(f"Total number of rows: {len(training_df.index)}")  # print total no of row to ensure it is correctly loaded
training_df.head(200)  # read the first 200 row of the training_df Dataframe
