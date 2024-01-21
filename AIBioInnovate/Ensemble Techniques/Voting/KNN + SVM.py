from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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

# Initialize the base models for voting
knn_clf = KNeighborsClassifier(n_neighbors=5)
svm_clf = SVC(probability=True, kernel='linear', random_state=42)

# Initialize Voting Classifier with the base models
voting_clf = VotingClassifier(estimators=[('knn', knn_clf), ('svm', svm_clf)], voting='soft')

# Train the voting classifier
voting_clf.fit(X_train, y_train)

# Evaluate the voting classifier
voting_predictions = voting_clf.predict(X_test)

# Print the performance metrics for the voting classifier
voting_accuracy = accuracy_score(y_test, voting_predictions)
voting_recall = recall_score(y_test, voting_predictions)
voting_f1 = f1_score(y_test, voting_predictions)

print(f"Voting Classifier Accuracy: {voting_accuracy}")
print(f"Voting Classifier Recall: {voting_recall}")
print(f"Voting Classifier F1 Score: {voting_f1}")

# Print classification report
print("Voting Classifier Report:")
print(classification_report(y_test, voting_predictions))