from django.shortcuts import render

# Create your views here.

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from .forms import MessageForm

email_dataset = pd.read_csv('C:/Users/user/Desktop/ml/Project/Spam Mail Prediction/mail_data.csv')
email_dataset.drop_duplicates(inplace=True)

# change Category string row into num row
label_encoder = LabelEncoder()
email_dataset['Category'] = label_encoder.fit_transform(email_dataset['Category'])

# split the dataset into feature and label
x = email_dataset['Message']
y = email_dataset['Category']

# vectorizer the feature col
vectorizer = TfidfVectorizer()
vectorizer.fit(x)
x = vectorizer.transform(x)
y = email_dataset['Category']

# separate the feature into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,  stratify=y, random_state=2)

# train a model
model = MultinomialNB()
model.fit(x_train, y_train)

# train prediction
train_prediction = model.predict(x_train)
train_accuracy = accuracy_score(y_train, train_prediction)


def predict_message(message):
    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)

    return "spam" if prediction[0] == 1 else "Ham"


def Home(request):
    result = None
    if request.method == 'POST':
        form = MessageForm(request.POST)
        if form.is_valid():
            message = form.cleaned_data['text']
            result = predict_message(message)
    else:
        form = MessageForm()

    return render(request, 'index.html', {'form': form, 'result': result})

