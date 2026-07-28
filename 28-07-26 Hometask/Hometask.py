import pandas as pd
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Download NLTK stopwords
nltk.download('stopwords')

# Load Dataset
data = pd.read_csv("IMDB Dataset.csv")

# Display first 5 rows
print("First 5 Records:")
print(data.head())

# Stopwords and Stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Text Preprocessing Function
def preprocess(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Tokenization
    words = text.split()

    # Remove stopwords and apply stemming
    words = [stemmer.stem(word)
             for word in words
             if word not in stop_words]

    return " ".join(words)

# Create cleaned text column
data["Clean_Text"] = data["review"].apply(preprocess)

print("\nCleaned Data:")
print(data[["review", "Clean_Text"]].head())

# TF-IDF Feature Extraction
tfidf = TfidfVectorizer()

X = tfidf.fit_transform(data["Clean_Text"])
y = data["sentiment"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Naive Bayes Model
model = MultinomialNB()

model.fit(X_train, y_train)

# Prediction
prediction = model.predict(X_test)

# Accuracy
print("\nAccuracy:")
print(accuracy_score(y_test, prediction))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, prediction))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, prediction))

# Predict New Review
new_review = [
    "This movie is fantastic and I really loved it."
]

# Preprocess the review
new_review_clean = [preprocess(review) for review in new_review]

# Convert to TF-IDF
new_text = tfidf.transform(new_review_clean)

# Predict
result = model.predict(new_text)

print("\nNew Review:")
print(new_review[0])

print("\nPredicted Sentiment:")
print(result[0])
