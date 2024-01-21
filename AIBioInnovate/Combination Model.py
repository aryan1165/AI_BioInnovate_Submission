from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import matplotlib.pyplot as plt
from collections import Counter

def calculate_diversity(sequence):
    # Calculate diversity (e.g., amino acid frequency) for a given sequence
    return dict(Counter(sequence))

def parse_fasta(file_path):
    # Parse .fasta file and return a list of sequences
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
    return sequences

def diversity_plot(sequences):
    # Calculate diversity for each sequence
    sequence_diversities = [calculate_diversity(seq) for seq in sequences]

    # Prepare data for the plot
    labels = list(sequence_diversities[0].keys())
    values = [[diversity.get(label, 0) for label in labels] for diversity in sequence_diversities]

    # Create a diversity plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, label in enumerate(labels):
        ax.plot(range(1, len(sequences) + 1), [value[i] for value in values], label=label)

    # Customize the plot
    ax.set_xlabel("Sequence Index")
    ax.set_ylabel("Frequency")
    ax.set_title("Diversity Plot")
    ax.legend()
    plt.show()

fasta_file_path = "moonlight.fasta"
sequences = parse_fasta(fasta_file_path)
diversity_plot(sequences)

fasta_file_path = "nonMP.fasta"
sequences = parse_fasta(fasta_file_path)
diversity_plot(sequences)



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
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def get_model_combinations(model_list, max_length):
    model_combinations = []
    for i in range(1, max_length + 1):
        model_combinations.extend(combinations(model_list, i))
    return model_combinations

base_models = [
    ('rf', RandomForestClassifier()),
    ('knn', KNeighborsClassifier()),
    ('svm', SVC(probability=True)),
    ('lr', LogisticRegression()),
    ('dt', DecisionTreeClassifier()),
    ('nb', GaussianNB())
]

final_estimators = [
    LogisticRegression(),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    SVC(probability=True),
    DecisionTreeClassifier(),
    GaussianNB()
]

def build_stacking_classifier(model_list, final_estimator):
    estimators = list(model_list)
    stacking_clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
    return stacking_clf

model_combinations = get_model_combinations(base_models, 5)
best_model = None
best_score = 0

for model_combo in model_combinations:
    for final_estimator in final_estimators:
        stacking_clf = build_stacking_classifier(model_combo, final_estimator)
        stacking_clf.fit(X_train_scaled, y_train)
        score = stacking_clf.score(X_test_scaled, y_test)
        
        if score > best_score:
            best_model = stacking_clf
            best_score = score

print(f"Best Model: {best_model}")
print(f"Best Score: {best_score}")

