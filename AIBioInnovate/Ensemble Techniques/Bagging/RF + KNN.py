from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score

# Function to parse FASTA files and extract sequences
def parse_fasta(file_path):
    fasta_sequences = SeqIO.parse(open(file_path), 'fasta')
    sequences = []
    for fasta in fasta_sequences:
        sequences.append(str(fasta.seq))
    return sequences

# Load the tokenizer and model from Hugging Face
model_name = "facebook/esm2_t12_35M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Parse the FASTA files
moonlighting_sequences = parse_fasta("moonlighting.fasta")
nonMP_sequences = parse_fasta("nonMP.fasta")

# Combine the sequences and create labels
sequences = moonlighting_sequences + nonMP_sequences
labels = [1] * len(moonlighting_sequences) + [0] * len(nonMP_sequences)

# Tokenize and encode sequences
inputs = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# Generate embeddings
with torch.no_grad():
    outputs = model(**inputs)

# Extract the embeddings
embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Initialize the base models for bagging
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
knn_clf = KNeighborsClassifier(n_neighbors=3)

# Initialize Bagging Classifiers with the base models
rf_bagging_clf = BaggingClassifier(base_estimator=rf_clf, n_estimators=10, random_state=42)
knn_bagging_clf = BaggingClassifier(base_estimator=knn_clf, n_estimators=10, random_state=42)

# Train the bagging models
rf_bagging_clf.fit(X_train, y_train)
knn_bagging_clf.fit(X_train, y_train)

# Evaluate the bagging models
rf_predictions = rf_bagging_clf.predict(X_test)
knn_predictions = knn_bagging_clf.predict(X_test)

# Print the performance metrics for each model
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions)

knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_recall = recall_score(y_test, knn_predictions)
knn_f1 = f1_score(y_test, knn_predictions)

print(f"Random Forest Bagging Classifier Accuracy: {rf_accuracy}")
print(f"Random Forest Bagging Classifier Recall: {rf_recall}")
print(f"Random Forest Bagging Classifier F1 Score: {rf_f1}")

print(f"KNN Bagging Classifier Accuracy: {knn_accuracy}")
print(f"KNN Bagging Classifier Recall: {knn_recall}")
print(f"KNN Bagging Classifier F1 Score: {knn_f1}")

# Print classification reports
print("Random Forest Bagging Classifier Report:")
print(classification_report(y_test, rf_predictions))

print("KNN Bagging Classifier Report:")
print(classification_report(y_test, knn_predictions))