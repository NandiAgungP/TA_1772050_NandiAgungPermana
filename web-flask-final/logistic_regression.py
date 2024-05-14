import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load the dataset
# Replace 'your_dataset.csv' with the actual filename.
data = pd.read_csv('data/imdb_dataset.csv')

# Preprocess the data
# Assuming the dataset has columns 'review' for text and 'sentiment' for labels.
reviews = data['review']
sentiments = data['sentiment']

# Convert text data to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to 5000 features
X = vectorizer.fit_transform(reviews)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, sentiments, test_size=0.2, random_state=42)

# Train the Logistic Regression classifier
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train, y_train)

# Evaluate the model
y_pred = log_reg.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model and vectorizer to .pkl files
with open('pkl/logistic_regression.pkl', 'wb') as model_file:
    pickle.dump(log_reg, model_file)

with open('pkl/vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("Model and vectorizer have been saved to 'logistic_regression.pkl' and 'vectorizer.pkl'")
