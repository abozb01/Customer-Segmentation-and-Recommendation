import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Generate synthetic data for customer segmentation
X = np.random.rand(100, 3)  # 100 customers with 3 features (e.g., age, income, purchase history)

# Perform customer segmentation using KMeans clustering
n_clusters = 3  # Number of customer segments
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
segment_labels = kmeans.fit_predict(X)

# Assume you have a target variable indicating whether the customer purchased a specific product (1) or not (0)
y = np.random.randint(2, size=100)  # Binary target variable for product purchase

# Split the data into training and testing sets for recommendation model
X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(X, y, segment_labels, test_size=0.2, random_state=42)

# Train a logistic regression model for each customer segment
models = {}
for label in range(n_clusters):
    # Filter the training data for the current segment
    X_train_segment = X_train[labels_train == label]
    y_train_segment = y_train[labels_train == label]
    
    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train_segment, y_train_segment)
    
    # Store the trained model for the current segment
    models[label] = model

# Make predictions for personalized product recommendations
recommendations = {}
for label, model in models.items():
    # Filter the testing data for the current segment
    X_test_segment = X_test[labels_test == label]
    
    # Make predictions using the trained model
    y_pred = model.predict(X_test_segment)
    
    # Store the predicted probabilities of purchasing the product
    recommendations[label] = model.predict_proba(X_test_segment)[:, 1]  # Probability of purchasing the product

# Convert probabilities to binary predictions using a threshold
binary_predictions = {label: (probs > 0.5).astype(int) for label, probs in recommendations.items()}

# Flatten the binary predictions and true labels for computing accuracy
flat_binary_predictions = np.concatenate(list(binary_predictions.values()))
flat_y_test = np.concatenate([y_test[labels_test == label] for label in binary_predictions.keys()])

# Compute accuracy
accuracy = accuracy_score(flat_y_test, flat_binary_predictions)
print("Recommendation Accuracy:", accuracy)

# Convert arrays to DataFrames and ensure they have the same length
max_length = max(len(probs) for probs in recommendations.values())
recommendations_df = pd.DataFrame({label: np.append(probs, [np.nan]*(max_length-len(probs))) for label, probs in recommendations.items()})

# Concatenate DataFrames
X_df = pd.DataFrame(X, columns=['Age', 'Income', 'Purchase_History'])
segment_labels_df = pd.DataFrame(segment_labels, columns=['Customer_Segment'])
y_test_df = pd.DataFrame(y_test, columns=['Purchased_Product'])
data_df = pd.concat([X_df, segment_labels_df, y_test_df, recommendations_df], axis=1)

# Modify the path to save the CSV file to the desktop
desktop_path = "~/Desktop/customer_data.csv"  # Replace "~" with your actual home directory if necessary

# Save DataFrame to CSV on the desktop with desired column names
data_df.to_csv(desktop_path, index=False)
