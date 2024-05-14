import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load the dataset
# Ensure your dataset is in a CSV file with columns such as 'review' and 'sentiment'.
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

# Train the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model and vectorizer to a .pkl file
with open('pkl/random_forest.pkl', 'wb') as f:
    pickle.dump(rf_classifier, f)

with open('pkl/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer have been saved to 'random_forest.pkl' and 'vectorizer.pkl'")
