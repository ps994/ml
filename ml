Naïve Bayes prediction model



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample dataset
emails = [
    "Win money now",          
    "Lowest price guaranteed",
    "Hey, are we still meeting tomorrow?",  
    "Let’s have lunch",       
    "Congratulations, you won!", 
    "Call me when you're free"
]
labels = [1, 1, 0, 0, 1, 0]  # 1 = spam, 0 = not spam

# Convert text data into numerical vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict on new unseen data
new_emails = ["You won a prize", "Let’s meet for dinner"]
new_X = vectorizer.transform(new_emails)
predictions = model.predict(new_X)

# Output predictions
print("Predictions:", predictions)  # 1 = spam, 0 = not spam




