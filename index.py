# Task 4: Machine Learning Model Implementation (CODTECH)

# Step 1: Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import warnings
warnings.filterwarnings('ignore')

# Step 2: Load Dataset
# Using a popular open-source dataset for spam detection
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# Step 3: Explore the Dataset
print("Dataset Shape:", df.shape)
print("\nSample Data:\n", df.head())

# Step 4: Data Cleaning and Preprocessing
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})  # Convert labels to numeric
print("\nLabel Distribution:\n", df['label'].value_counts())

# Visualize the class distribution
sns.countplot(x='label', data=df)
plt.title("Spam vs Ham Distribution")
plt.show()

# Step 5: Splitting Data into Train and Test Sets
X = df['message']
y = df['label_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Creating a Pipeline
# This pipeline performs: Text Vectorization + TF-IDF Transformation + Classification
pipeline = Pipeline([
    ('vect', CountVectorizer()),       # Convert text to word count vectors
    ('tfidf', TfidfTransformer()),     # Apply TF-IDF
    ('clf', MultinomialNB()),          # Naive Bayes classifier
])

# Step 7: Train the Model
pipeline.fit(X_train, y_train)

# Step 8: Make Predictions
y_pred = pipeline.predict(X_test)

# Step 9: Model Evaluation
print("\nModel Evaluation:\n")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 10: Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Step 11: Testing with Custom Inputs
sample_messages = [
    "Congratulations! You've won a free ticket to Bahamas. Text WIN to 12345",
    "Hey, are we still meeting for coffee today?",
    "URGENT! Your account has been compromised. Send your password to reset."
]

sample_preds = pipeline.predict(sample_messages)

for msg, pred in zip(sample_messages, sample_preds):
    print(f"\nMessage: {msg}\nPrediction: {'Spam' if pred == 1 else 'Ham'}")

# End of Task
print("\nTask Completed Successfully.")
