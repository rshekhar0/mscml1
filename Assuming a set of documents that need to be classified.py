from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd

def naive_bayes_classification(documents, labels):
    # Convert text data to a matrix of token counts
    vectorizer = CountVectorizer()
    X_counts = vectorizer.fit_transform(documents)

    # Train a Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(X_counts, labels)

    return vectorizer, clf

def classify_documents(new_documents, vectorizer, clf):
    # Convert the new text data to a matrix of token counts
    X_new_counts = vectorizer.transform(new_documents)

    # Make predictions on the new set of documents
    predictions = clf.predict(X_new_counts)

    return predictions

# Sample documents and labels
documents = [
    "This is a positive document.",
    "Negative sentiment is observed here.",
    "I feel positive about this.",
    "I do not like this at all.",
    "This is a negative example.",
]

labels = ["Positive", "Negative", "Positive", "Negative", "Negative"]

# Train the classifier
vectorizer, clf = naive_bayes_classification(documents, labels)

# New set of documents to be classified
new_documents = [
    "This document contains positive sentiment.",
    "Negative emotions are evident in this text.",
    "I am feeling optimistic about the outcome.",
    "This document expresses dissatisfaction.",
    "Positive examples uplift the mood.",
]

# Classify the new set of documents
predictions = classify_documents(new_documents, vectorizer, clf)

# Display the predictions
for doc, label in zip(new_documents, predictions):
    print(f"Document: {doc} \nPredicted Label: {label}\n")

# Plot the workflow diagram
def plot_workflow():
    plt.figure(figsize=(8, 4))
    plt.text(0.5, 0.5, "Data Preparation\nand Splitting", ha='center', va='center', fontsize=12)
    plt.text(0.5, 0.3, "Feature Extraction\n(Count Vectorization)", ha='center', va='center', fontsize=12)
    plt.text(0.5, 0.1, "Model Training\n(Naive Bayes)", ha='center', va='center', fontsize=12)
    plt.arrow(0.2, 0.5, 0.1, 0, head_width=0.05, head_length=0.05, fc='black', ec='black')
    plt.arrow(0.5, 0.3, 0, 0.1, head_width=0.05, head_length=0.05, fc='black', ec='black')
    plt.arrow(0.5, 0.1, 0, 0.1, head_width=0.05, head_length=0.05, fc='black', ec='black')
    plt.axis('off')
    plt.title('Workflow Diagram')
    plt.show()

plot_workflow()

# Display summary report
def generate_summary_report(true_labels, predicted_labels):
    report = classification_report(true_labels, predicted_labels, output_dict=True)
    summary_df = pd.DataFrame(report).transpose()
    return summary_df

summary_df = generate_summary_report(labels, clf.predict(vectorizer.transform(documents)))
print("\nSummary Report:")
print(summary_df)
