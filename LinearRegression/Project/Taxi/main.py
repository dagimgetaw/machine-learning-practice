import pandas as pd  # use for data manipulation and analysis

# load the csv file from the given URL and store as Dataframe
chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")
# select specific colum we want and store as Dataframe
training_df = chicago_taxi_dataset[['TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE']]
print(f"Total number of rows: {len(training_df.index)}")  # print total no of row to ensure it is correctly loaded
training_df.head(200)  # read the first 200 row of the training_df Dataframe

# training_df.describe(include='all')  # this means we use all row the analysis

# To get the maximum amount of value from 'FARE' column
max_fare = training_df['FARE'].max()
print(f"The maximum fare: {max_fare:.2f}")

# To get the mean distance across all trips from 'TRIP_MILES' column
mean_distance = training_df['TRIP_MILES'].mean()
print(f"The mean distance across all trips: {mean_distance:.4f}")

# To get unique companies in the dataset from 'COMPANY' column
num_unique_companies = training_df['COMPANY'].nunique()
print(f"Unique companies in the dataset: {num_unique_companies}")

# To get most frequent payment type from 'PAYMENT_TYPE' column
most_freq_payment_type = training_df['PAYMENT_TYPE'].value_counts().idxmax()
print(f"Most frequent payment type: {most_freq_payment_type}")

# Are any features missing data
missing_values = training_df.isnull().sum().sum()
print("Are any features missing data: ", "No" if missing_values == 0 else "Yes")

# Generate a correlation matrix
correlation_matrix = training_df.corr(numeric_only=True)
print("\nCorrelation matrix:\n", correlation_matrix)

# Find the strongest and weakest correlation with 'FARE'
strongest_corr_feature = correlation_matrix['FARE'].drop('FARE').idxmax()
weakest_corr_feature = correlation_matrix['FARE'].drop('FARE').idxmin()

print(f"\nFeature that correlates most strongly with FARE: {strongest_corr_feature}")
print(f"Feature that correlates least strongly with FARE: {weakest_corr_feature}")
