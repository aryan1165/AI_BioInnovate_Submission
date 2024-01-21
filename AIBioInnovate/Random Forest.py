from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
model = AutoModel.from_pretrained(model_name).to(device)

# Parse the FASTA files
moonlighting_sequences = parse_fasta("moonlighting.fasta")
nonMP_sequences = parse_fasta("nonMP.fasta")

# Combine the sequences and create labels
sequences = moonlighting_sequences + nonMP_sequences
labels = [1] * len(moonlighting_sequences) + [0] * len(nonMP_sequences)

# Tokenize and encode sequences in batches
batch_size = 16  # Adjust batch size according to your GPU memory
embeddings = []
for i in range(0, len(sequences), batch_size):
    batch = sequences[i:i+batch_size]
    inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the embeddings and move to CPU
    batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    embeddings.append(batch_embeddings)

# Convert the list of embeddings into a NumPy array
embeddings = np.vstack(embeddings)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Define the classifier
rf_clf = RandomForestClassifier(class_weight='balanced')

# Train the model
rf_clf.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Print the accuracy, recall, and F1 score
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")