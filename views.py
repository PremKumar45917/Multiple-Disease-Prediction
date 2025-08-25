from django.shortcuts import render, redirect
from .models import ModelResults
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('media/disease_dataset.csv')
X = df.drop('Disease', axis=1)   # ✅ Fixed
y = df['Disease']                # ✅ Fixed

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Helper: train and store results
def train_model(model, name):
    try:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        acc = round(accuracy_score(y_test, y_pred), 3)
        prec = round(precision_score(y_test, y_pred, average='macro'), 3)
        rec = round(recall_score(y_test, y_pred, average='macro'), 3)
        f1 = round(f1_score(y_test, y_pred, average='macro'), 3)

        ModelResults.objects.create(
            model_name=name,
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1_score=f1
        )
    except Exception as e:
        print(f"❌ Error training {name}: {e}")

# Training view
def training(request):
    # Clear previous results
    ModelResults.objects.all().delete()

    # Train all models
    models = [
        (DecisionTreeClassifier(), "Decision Tree"),
        (RandomForestClassifier(), "Random Forest"),
        (BernoulliNB(), "Naive Bayes"),
        (SVC(), "Support Vector Classifier"),
        (KNeighborsClassifier(), "KNN"),
        (LogisticRegression(max_iter=1000), "Logistic Regression"),
    ]

    for model, name in models:
        train_model(model, name)

    results = ModelResults.objects.all()
    return render(request, 'users/modelresults.html', {'results_list': results})
