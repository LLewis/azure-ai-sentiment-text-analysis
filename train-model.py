import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load the data
data = pd.read_csv('review-data.csv')

# Feature extraction: For simplicity, we'll use year and rating as features
data['is_positive'] = data['rating'] >= 4  # Consider ratings 4 and 5 as positive

# Prepare data for modeling
X = data[['year', 'rating']]
y = data['product']

# Split the data  --> data to train the model and test data to use with new trained model
# random_state= 42,  set to produce the same results across a different run.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the nodel
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model - allows to store mnodel on disk and load/use without retraining it
joblib.dump(model, 'review_model.pkl')
