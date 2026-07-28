import pandas as pd
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download stopwords
nltk.download('stopwords')

# Load dataset
data = pd.read_csv("IMDB Dataset.csv")
# or data = pd.read_csv("sentiment.csv")

print(data.head())

# Stopwords
stop_words = set(stopwords.words('english'))

# Stemmer
stemmer = PorterStemmer()

# Preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Dataset columns are 'review' and 'sentiment'
data["Clean_Text"] = data["review"].apply(preprocess)

print(data[["review", "Clean_Text"]].head())

# TF-IDF
tfidf = TfidfVectorizer()

X = tfidf.fit_transform(data["Clean_Text"])
y = data["sentiment"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train
model = MultinomialNB()
model.fit(X_train, y_train)

# Test
prediction = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, prediction))

print("\nClassification Report")
print(classification_report(y_test, prediction))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, prediction))

# Predict new review
new_review = ["This movie is fantastic and I really loved it"]

new_review = [preprocess(review) for review in new_review]

new_text = tfidf.transform(new_review)

result = model.predict(new_text)

print("\nPrediction:", result)
