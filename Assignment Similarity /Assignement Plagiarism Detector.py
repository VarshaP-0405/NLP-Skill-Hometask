!pip install -q rarfile
!apt-get update -qq
!apt-get install -y -qq unrar

import os
import re
import zipfile
import rarfile
import pandas as pd
from google.colab import files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

uploaded = files.upload()
dataset_file = list(uploaded.keys())[0]

extract_path = "/content/pan_dataset"
os.makedirs(extract_path, exist_ok=True)

if dataset_file.lower().endswith(".zip"):
    with zipfile.ZipFile(dataset_file, "r") as zip_ref:
        zip_ref.extractall(extract_path)
elif dataset_file.lower().endswith(".rar"):
    with rarfile.RarFile(dataset_file) as rar_ref:
        rar_ref.extractall(extract_path)

txt_files = []

for root, dirs, files_list in os.walk(extract_path):
    for file in files_list:
        if file.lower().endswith(".txt"):
            txt_files.append(os.path.join(root, file))

print("Total text files found:", len(txt_files))

MAX_DOCUMENTS = 10
selected_files = txt_files[:MAX_DOCUMENTS]

documents = []
names = []

for file in selected_files:
    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    if text.strip():
        documents.append(text)
        names.append(os.path.basename(file))

print("Documents loaded:", len(documents))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

cleaned_documents = [clean_text(text) for text in documents]

tokens = [text.split() for text in cleaned_documents]

for i in range(min(3, len(tokens))):
    print(names[i], ":", tokens[i][:20])

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(cleaned_documents)

print("TF-IDF matrix shape:", tfidf_matrix.shape)

similarity_matrix = cosine_similarity(tfidf_matrix)
similarity_percentage = similarity_matrix * 100

similarity_df = pd.DataFrame(
    similarity_percentage,
    index=names,
    columns=names
)

print("\nSimilarity Matrix:")
display(similarity_df.round(2))

threshold = 70

results = []

for i in range(len(names)):
    for j in range(i + 1, len(names)):
        score = similarity_percentage[i][j]

        if score >= threshold:
            status = "Possible Plagiarism"
        else:
            status = "Low Similarity"

        results.append({
            "Document 1": names[i],
            "Document 2": names[j],
            "Similarity (%)": round(score, 2),
            "Status": status
        })

results = sorted(
    results,
    key=lambda x: x["Similarity (%)"],
    reverse=True
)

report = pd.DataFrame(results)

print("\nSimilarity Report:")
display(report)

print("\nRanked Document Pairs:")

for i, result in enumerate(results, 1):
    print(
        f"{i}. {result['Document 1']} vs "
        f"{result['Document 2']} : "
        f"{result['Similarity (%)']}% -> "
        f"{result['Status']}"
    )

suspicious = report[
    report["Similarity (%)"] >= threshold
]

print("\nSuspicious Document Pairs:")

if len(suspicious) > 0:
    display(suspicious)
else:
    print("No document pairs crossed the plagiarism threshold.")

similarity_df.round(2).to_csv(
    "/content/similarity_matrix.csv"
)

report.to_csv(
    "/content/plagiarism_report.csv",
    index=False
)

files.download("/content/similarity_matrix.csv")
files.download("/content/plagiarism_report.csv")
