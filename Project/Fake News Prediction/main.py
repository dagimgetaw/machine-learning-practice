import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

news_dataset = pd.read_csv('train.csv')
news_dataset = news_dataset.fillna('')

news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

x = news_dataset.drop(columns='label', axis=1)
y = news_dataset['label']

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


news_dataset['content'] = news_dataset['content'].apply(stemming)
x = news_dataset['content']
y = news_dataset['label']

vectorizer = TfidfVectorizer()
vectorizer.fit(x)
x = vectorizer.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify=y, random_state=2)
model = LogisticRegression()
model.fit(x_train, y_train)

train_prediction = model.predict(x_train)
train_accuracy = accuracy_score(y_train, train_prediction)
print(f"The training accuracy is {train_accuracy}")

test_prediction = model.predict(x_test)
test_accuracy = accuracy_score(y_test, test_prediction)
print(f"The testing accuracy is {test_accuracy}")

# Load the test dataset
test_dataset = pd.read_csv('test.csv')
test_dataset = test_dataset.fillna('')

# Create the 'content' column as in training data
test_dataset['content'] = test_dataset['author'] + ' ' + test_dataset['title']

# Apply the same preprocessing function
test_dataset['content'] = test_dataset['content'].apply(stemming)

# Transform the test data using the same vectorizer
x_test_final = vectorizer.transform(test_dataset['content'])

# Make predictions
test_predictions = model.predict(x_test_final)

# Load submit.csv
submit_dataset = pd.read_csv('submit.csv')

# Compare predictions with submit.csv
correct_predictions = sum(test_predictions == submit_dataset['label'])
total_samples = len(submit_dataset)
accuracy = correct_predictions / total_samples

print(f"Model accuracy on test.csv compared to submit.csv: {accuracy * 100:.2f}%")
