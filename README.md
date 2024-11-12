# Moonlight Protein Engineering - AI BioInnovate Submission

## Overview

This project, focuses on applying machine learning and deep learning techniques to analyze and engineer moonlight proteins. Leveraging Meta's ESM-2 model, this project generates embedding sequences from protein sequences and performs extensive comparative analysis across various machine learning classifiers, including Bi-LSTM, MLP, and ensemble methods.

### Key Highlights
- **Embedding Generation**: Used Meta's ESM-2 model to generate embedding sequences for moonlight proteins.
- **Classifier Comparison**: Evaluated performance across several models including Bi-LSTM, MLP, Logistic Regression, K-Nearest Neighbors, Random Forest, Gaussian Naive Bayes, Support Vector Machines, and Decision Trees.
- **Ensemble Techniques**: Enhanced accuracy by 7.2%, achieving a peak accuracy of ~81.7% through ensemble methods.

---

## Project Structure

### Main Directory

- **`Bi-LSTM.py`**: Implements a Bidirectional LSTM model for sequence classification.
- **`Combination Model.py`**: Combines different models or features for enhanced performance.
- **`KNN Classifiers.py`**: K-Nearest Neighbors classifier implementation.
- **`MLP (Multilayer Perceptron).py`**: Multilayer Perceptron model for protein sequence classification.
- **`Random Forest.py`**: Random Forest classifier for protein sequences.
- **`SVC.py`**: Support Vector Classifier.
- **`moonlight.fasta`**: FASTA file containing moonlight protein sequences.
- **`nonMP.fasta`**: FASTA file with non-moonlight protein sequences.

### Ensemble Techniques

Contains directories for various ensemble learning techniques:

- **Bagging**: Implements Bagging techniques for aggregating classifiers.
- **Boosting**: Boosting models to enhance prediction performance.
- **Stacking**: Combines multiple model predictions using stacking techniques.
- **Voting**: Voting classifiers that aggregate predictions from various models.


## Results

The comparative analysis demonstrated the highest accuracy with ensemble techniques, achieving an accuracy of **~81.7%**. The ensembling approach improved the accuracy by **7.2%**, showcasing its efficacy in protein classification tasks.

---


