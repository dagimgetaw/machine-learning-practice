import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Load Data
train_dataset = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
test_dataset = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

# Drop Duplicates
train_dataset.drop_duplicates(inplace=True)
test_dataset.drop_duplicates(inplace=True)

# Handle Missing Values
imputer = SimpleImputer(strategy='median')
train_dataset[['age', 'fare']] = imputer.fit_transform(train_dataset[['age', 'fare']])
test_dataset[['age', 'fare']] = imputer.transform(test_dataset[['age', 'fare']])

# Encode Categorical Variables
label_encoder = LabelEncoder()
for col in ['sex', 'alone']:
    train_dataset[col] = label_encoder.fit_transform(train_dataset[col])
    test_dataset[col] = label_encoder.transform(test_dataset[col])

# One-Hot Encoding for 'class'
train_dataset = pd.get_dummies(train_dataset, columns=['class'], drop_first=True)
test_dataset = pd.get_dummies(test_dataset, columns=['class'], drop_first=True)

# Feature Engineering
for dataset in [train_dataset, test_dataset]:
    dataset['family size'] = dataset['n_siblings_spouses'] + dataset['parch']
    dataset['is_child'] = (dataset['age'] < 10).astype(int)
    dataset['fare_per_person'] = dataset['fare'] / (dataset['family size'] + 1)

# Define Features & Target
x_train = train_dataset.drop(columns=['survived', 'n_siblings_spouses', 'parch', 'deck', 'embark_town'])
y_train = train_dataset['survived']

x_test = test_dataset.drop(columns=['survived', 'n_siblings_spouses', 'parch', 'deck', 'embark_town'])
y_test = test_dataset['survived']

# Standardization
scaler = StandardScaler()
x_train[x_train.columns] = scaler.fit_transform(x_train)
x_test[x_test.columns] = scaler.transform(x_test)

# Import Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Initialize Models
models = {
    'LogReg': LogisticRegression(max_iter=200),
    'DecisionTree': DecisionTreeClassifier(max_depth=10, min_samples_split=5),
    'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10),
    'AdaBoost': AdaBoostClassifier(n_estimators=100),
    'GradientBoost': GradientBoostingClassifier(n_estimators=200),
    'SVC': SVC(kernel='rbf', C=1.0),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'NaiveBayes': GaussianNB(),
    'Perceptron': Perceptron(),
    'LinearSVC': LinearSVC(),
    'SGDClassifier': SGDClassifier(),
    'MLPClassifier': MLPClassifier(),
    'XGBoost': xgb.XGBClassifier(n_estimators=200, learning_rate=0.001),
    'LightGBM': lgb.LGBMClassifier(n_estimators=200),
    'CatBoost': CatBoostClassifier(iterations=1000, depth=6, silent=True)
}

# Train & Evaluate Models
best_accuracy = 0
best_model = None
best_model_name = ""

for model_name, model in models.items():
    model.fit(x_train, y_train)
    model_prediction = model.predict(x_test)
    model_accuracy = accuracy_score(y_test, model_prediction)

    if model_accuracy > best_accuracy:
        best_accuracy = model_accuracy
        best_model = model
        best_model_name = model_name

    print(f'{model_name} Accuracy: {model_accuracy:.4f}')

# Final Evaluation on Test Dataset
final_predictions = best_model.predict(x_test)
final_accuracy = accuracy_score(y_test, final_predictions)

print(f"\nBest Model: {best_model_name} with Training Accuracy: {best_accuracy:.4f}")
print(f"Final Accuracy on Test Dataset: {final_accuracy:.4f}")
