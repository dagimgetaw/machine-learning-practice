import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, auc, f1_score
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import sklearn.metrics as metrics

loan_dataset = pd.read_csv('loan_approval_dataset.csv')

loan_dataset.drop_duplicates(inplace=True)

loan_dataset.columns = loan_dataset.columns.str.replace(' ', '')
loan = loan_dataset.drop(['loan_id'], axis=1)
sns.pairplot(loan)

sns.scatterplot(x=loan['cibil_score'], y= loan['loan_amount'], hue=loan['loan_status'])
plt.title("Loan Status, Loan Amount, Credit Score")
plt.xlabel("Credit Score")
plt.ylabel("Loan Amount")
plt.show()

label_encoder = LabelEncoder()

loan_dataset['education'] = label_encoder.fit_transform(loan_dataset['education'])
loan_dataset['self_employed'] = label_encoder.fit_transform(loan_dataset['self_employed'])
loan_dataset['loan_status'] = label_encoder.fit_transform(loan_dataset['loan_status'])

scaler = StandardScaler()

columns_to_scale = [
    'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
    'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
]

loan_dataset[columns_to_scale] = scaler.fit_transform(loan_dataset[columns_to_scale])

loan_dataset.drop(columns=['loan_id'], axis=1)


x = loan_dataset.drop(columns=['loan_status'], axis=1)
y = loan_dataset['loan_status']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

param_dist = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier()
random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy')
random_search.fit(x_train, y_train)

best_params = random_search.best_params_
best_forest = random_search.best_estimator_

rf_opt = RandomForestClassifier(n_estimators = 100, max_depth = 20,
                                min_samples_leaf = 1, min_samples_split = 2,random_state = 0)
rf_opt.fit(x_train, y_train)
y_rf = rf_opt.predict(x_test)

y_test_rf = rf_opt.predict(x_test)

print('Accuracy:', '%.3f' % accuracy_score(y_test, y_test_rf))
print('Precision:', '%.3f' % precision_score(y_test, y_test_rf))
print('Recall:', '%.3f' % recall_score(y_test, y_test_rf))
print('F1 Score:', '%.3f' % f1_score(y_test, y_test_rf))

cm = metrics.confusion_matrix(y_test, y_rf, labels = rf_opt.classes_)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm,display_labels = rf_opt.classes_)
disp.plot()
