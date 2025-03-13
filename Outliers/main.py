import pandas as pd

wh_dataset = pd.read_csv("weight-height.xls")

# Calculate Quartiles
H_Q1 = wh_dataset.Height.quantile(0.25)
H_Q3 = wh_dataset.Height.quantile(0.75)
W_Q1 = wh_dataset.Weight.quantile(0.25)
W_Q3 = wh_dataset.Weight.quantile(0.75)

# Calculate IQR (Interquartile Range)
H_IQR = H_Q3 - H_Q1
W_IQR = W_Q3 - W_Q1

# Compute Outlier Limits
h_lower_limit = H_Q1 - 1.5 * H_IQR
h_higher_limit = H_Q3 + 1.5 * H_IQR

w_lower_limit = W_Q1 - 1.5 * W_IQR
w_higher_limit = W_Q3 + 1.5 * W_IQR

# Find Outliers
outliers_height = wh_dataset[(wh_dataset.Height < h_lower_limit) | (wh_dataset.Height > h_higher_limit)]
outliers_weight = wh_dataset[(wh_dataset.Weight < w_lower_limit) | (wh_dataset.Weight > w_higher_limit)]

outliers = pd.concat([outliers_height, outliers_weight]).drop_duplicates()

# Find Data Without Outliers
no_outlier_data = wh_dataset[(wh_dataset.Height >= h_lower_limit) & (wh_dataset.Height <= h_higher_limit) &
                             (wh_dataset.Weight >= w_lower_limit) & (wh_dataset.Weight <= w_higher_limit)]

# Save Outliers to CSV
outliers.to_csv("outliers.csv", index=False)

# Display the number of outliers
print("Number of Outliers:", len(outliers))
