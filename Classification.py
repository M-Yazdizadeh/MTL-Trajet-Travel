import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# -----------------------------------
# Step 1: Load the Dataset
# -----------------------------------
# Define the path to the CSV file containing the first 100,000 rows
file_path = 'first_100000_rows.csv'

# Use Pandas to read the CSV file into a DataFrame
data = pd.read_csv(file_path)

# -----------------------------------
# Step 2: Preprocessing
# -----------------------------------
# Drop unnecessary columns that are not needed for the analysis or modeling
# 'id_trip', 'geometry', 'timestamp', 'starttime', and 'endtime' are removed
data = data.drop(columns=['id_trip', 'geometry', 'timestamp', 'starttime', 'endtime'])

# Translate and simplify French categories to English for better readability and consistency
translations = {
    'Voiture / Moto': 'Car / Motorbike',
    'Transport collectif': 'Public Transport',
    'Vélo': 'Bicycle',
    'À pied': 'Walking',
    'À pied, Transport collectif': 'Walking + Public Transport',
    'À pied, Transport collectif, Vélo': 'Walking + Public Transport + Bicycle',
    'À pied, Voiture / Moto, Vélo': 'Walking + Car / Motorbike + Bicycle',
    'À pied, Vélo': 'Walking + Bicycle',
    'Transport collectif, Voiture / Moto': 'Public Transport + Car / Motorbike'
}
# Apply the translations to the 'mode' column
data['mode'] = data['mode'].map(translations)

# Handle missing target values by dropping rows where 'mode' is NaN after translation
data = data.dropna(subset=['mode'])

# Encode categorical variables using Label Encoding
# Label Encoding converts categorical labels into numeric form
le_mode = LabelEncoder()
data['mode'] = le_mode.fit_transform(data['mode'])

# If the 'purpose' column exists, encode it as well
if 'purpose' in data.columns:
    le_purpose = LabelEncoder()
    # Fill missing values in 'purpose' with 'Unknown' before encoding
    data['purpose'] = le_purpose.fit_transform(data['purpose'].fillna('Unknown'))

# Separate features (X) and target variable (y)
X = data.drop(columns=['mode'])
y = data['mode']

# Handle missing values in features using SimpleImputer
# Strategy 'mean' replaces missing values with the mean of the column
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Address class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
# SMOTE generates synthetic samples for minority classes to balance the dataset
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Split the data into training and testing sets
# Test size is 20% of the dataset, and random_state ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features for models that are sensitive to feature scaling (e.g., MLP and Deep Learning)
# StandardScaler standardizes features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------
# Step 3: PyTorch Deep Learning Model
# -----------------------------------
# Define a custom dataset class for PyTorch
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        """
        Initialize the dataset with features and labels.
        Convert features to float tensors and labels to long tensors.
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.long)  # Ensure labels are NumPy array

    def __len__(self):
        """
        Return the total number of samples.
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        Retrieve a single sample at the specified index.
        """
        return self.features[idx], self.labels[idx]

# Define the neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        """
        Initialize the neural network layers.
        - input_size: Number of input features
        - num_classes: Number of output classes
        """
        super(NeuralNetwork, self).__init__()
        # First fully connected layer with 128 neurons
        self.fc1 = nn.Linear(input_size, 128)
        # Dropout layer to prevent overfitting
        self.dropout1 = nn.Dropout(0.3)
        # ReLU activation function introduces non-linearity
        self.relu = nn.ReLU()
        # Second fully connected layer with 64 neurons
        self.fc2 = nn.Linear(128, 64)
        # Another Dropout layer
        self.dropout2 = nn.Dropout(0.3)
        # Output layer with neurons equal to the number of classes
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        Define the forward pass through the network.
        """
        # Pass input through first layer, apply ReLU, then dropout
        x = self.dropout1(self.relu(self.fc1(x)))
        # Pass through second layer, apply ReLU, then dropout
        x = self.dropout2(self.relu(self.fc2(x)))
        # Output layer (no activation here; handled by loss function)
        x = self.fc3(x)
        return x

# Prepare data for PyTorch by creating dataset and dataloader instances
train_dataset = CustomDataset(X_train_scaled, y_train)
test_dataset = CustomDataset(X_test_scaled, y_test)

# DataLoader for batching and shuffling the training data
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# DataLoader for batching the testing data
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the PyTorch model
input_size = X_train.shape[1]  # Number of features
num_classes = len(np.unique(y))  # Number of unique target classes
model = NeuralNetwork(input_size, num_classes)

# Define the loss function as Cross Entropy Loss for multi-class classification
criterion = nn.CrossEntropyLoss()
# Define the optimizer as Adam with a learning rate of 0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Learning rate scheduler decreases the learning rate by a factor of 0.5 every 5 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training loop for the neural network
epochs = 20  # Number of epochs to train
for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Clear the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        running_loss += loss.item()  # Accumulate loss
    scheduler.step()  # Update learning rate
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

# Evaluation of the neural network on the test set
model.eval()  # Set model to evaluation mode
all_preds, all_labels = [], []
with torch.no_grad():  # Disable gradient computation
    for inputs, labels in test_loader:
        outputs = model(inputs)  # Forward pass
        _, preds = torch.max(outputs, 1)  # Get predictions
        all_preds.extend(preds.numpy())  # Collect predictions
        all_labels.extend(labels.numpy())  # Collect true labels

print("\n=== PyTorch Neural Network ===")
print("Classification Report:")
# Generate and print the classification report
print(classification_report(all_labels, all_preds, target_names=le_mode.classes_))

# -----------------------------------
# Step 4: Improved Decision Tree Classifier
# -----------------------------------
print("\n=== Decision Tree Classifier ===")
# Define the hyperparameter grid for Decision Tree
dt_params = {
    'max_depth': [10, 20, None],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'class_weight': ['balanced', None]  # Weights associated with classes
}

# Initialize Decision Tree Classifier with a fixed random state for reproducibility
dt_clf = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_params,
    n_iter=10,  # Number of parameter settings that are sampled
    cv=3,  # 3-fold cross-validation
    random_state=42
)

# Fit the model on the training data
dt_clf.fit(X_train, y_train)

# Predict on the test data
dt_y_pred = dt_clf.predict(X_test)

# Print the best hyperparameters found
print("Best Parameters:", dt_clf.best_params_)
# Print the classification report
print("Classification Report:")
print(classification_report(y_test, dt_y_pred, target_names=le_mode.classes_))

# -----------------------------------
# Step 5: Improved Random Forest Classifier
# -----------------------------------
print("\n=== Random Forest Classifier ===")
# Define the hyperparameter grid for Random Forest
rf_params = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [10, 20, None],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'class_weight': ['balanced']  # Weights associated with classes
}

# Initialize Random Forest Classifier with a fixed random state for reproducibility
rf_clf = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    n_iter=10,  # Number of parameter settings that are sampled
    cv=3,  # 3-fold cross-validation
    random_state=42
)

# Fit the model on the training data
rf_clf.fit(X_train, y_train)

# Predict on the test data
rf_y_pred = rf_clf.predict(X_test)

# Print the best hyperparameters found
print("Best Parameters:", rf_clf.best_params_)
# Print the classification report
print("Classification Report:")
print(classification_report(y_test, rf_y_pred, target_names=le_mode.classes_))

# -----------------------------------
# Step 6: Improved XGBoost Classifier
# -----------------------------------
print("\n=== Gradient Boosting (XGBoost) ===")
# Define the hyperparameter grid for XGBoost
xgb_params = {
    'n_estimators': [100, 200, 300],  # Number of gradient boosted trees
    'max_depth': [10, 20, 30],  # Maximum tree depth for base learners
    'learning_rate': [0.01, 0.1, 0.2],  # Boosting learning rate (eta)
    'subsample': [0.8, 0.9, 1.0],  # Subsample ratio of the training instances
    'colsample_bytree': [0.8, 0.9, 1.0]  # Subsample ratio of columns when constructing each tree
}

# Initialize XGBoost Classifier with a fixed random state for reproducibility
xgb_clf = RandomizedSearchCV(
    XGBClassifier(eval_metric='mlogloss', random_state=42),
    xgb_params,
    n_iter=10,  # Number of parameter settings that are sampled
    cv=3,  # 3-fold cross-validation
    random_state=42
)

# Fit the model on the training data
xgb_clf.fit(X_train, y_train)

# Predict on the test data
xgb_y_pred = xgb_clf.predict(X_test)

# Print the best hyperparameters found
print("Best Parameters:", xgb_clf.best_params_)
# Print the classification report
print("Classification Report:")
print(classification_report(y_test, xgb_y_pred, target_names=le_mode.classes_))

# -----------------------------------
# Step 7: Improved Neural Network (MLP)
# -----------------------------------
print("\n=== Neural Network (MLP) ===")
# Define the hyperparameter grid for MLPClassifier
mlp_params = {
    'hidden_layer_sizes': [(128, 64), (128, 64, 32), (256, 128)],  # Sizes of hidden layers
    'alpha': [0.0001, 0.001, 0.01],  # L2 penalty (regularization term) parameter
    'learning_rate': ['constant', 'adaptive'],  # Learning rate schedule for weight updates
    'max_iter': [400, 500]  # Maximum number of iterations
}

# Initialize MLPClassifier with a fixed random state for reproducibility
mlp_clf = RandomizedSearchCV(
    MLPClassifier(random_state=42),
    mlp_params,
    n_iter=10,  # Number of parameter settings that are sampled
    cv=3,  # 3-fold cross-validation
    random_state=42
)

# Fit the model on the scaled training data
mlp_clf.fit(X_train_scaled, y_train)

# Predict on the scaled test data
mlp_y_pred = mlp_clf.predict(X_test_scaled)

# Print the best hyperparameters found
print("Best Parameters:", mlp_clf.best_params_)
# Print the classification report
print("Classification Report:")
print(classification_report(y_test, mlp_y_pred, target_names=le_mode.classes_))







