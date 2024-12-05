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

# Load the dataset
file_path = 'first_100000_rows.csv'
data = pd.read_csv(file_path)

# Preprocessing
# Drop unnecessary columns
data = data.drop(columns=['id_trip', 'geometry', 'timestamp', 'starttime', 'endtime'])

# Translate and simplify French categories to English
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
data['mode'] = data['mode'].map(translations)

# Handle missing target values
data = data.dropna(subset=['mode'])

# Encode categorical variables
le_mode = LabelEncoder()
data['mode'] = le_mode.fit_transform(data['mode'])

if 'purpose' in data.columns:
    le_purpose = LabelEncoder()
    data['purpose'] = le_purpose.fit_transform(data['purpose'].fillna('Unknown'))

# Separate features and target
X = data.drop(columns=['mode'])
y = data['mode']

# Handle missing values in features
imputer = SimpleImputer(strategy='mean')  # Replace NaN with mean
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Address class imbalance with SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features for MLP and Deep Learning
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- PyTorch Deep Learning Model ---
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.long)  # Ensure labels are NumPy array

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# Prepare data for PyTorch
train_dataset = CustomDataset(X_train_scaled, y_train)
test_dataset = CustomDataset(X_test_scaled, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize PyTorch model
input_size = X_train.shape[1]
num_classes = len(np.unique(y))
model = NeuralNetwork(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training loop
epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

# Evaluation
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

print("\n=== PyTorch Neural Network ===")
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=le_mode.classes_))

# --- 1. Improved Decision Tree Classifier ---
print("\n=== Decision Tree Classifier ===")
dt_params = {
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None]
}
dt_clf = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_params,
    n_iter=10,
    cv=3,
    random_state=42
)
dt_clf.fit(X_train, y_train)
dt_y_pred = dt_clf.predict(X_test)
print("Best Parameters:", dt_clf.best_params_)
print("Classification Report:")
print(classification_report(y_test, dt_y_pred, target_names=le_mode.classes_))

# --- 2. Improved Random Forest Classifier ---
print("\n=== Random Forest Classifier ===")
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced']
}
rf_clf = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    n_iter=10,
    cv=3,
    random_state=42
)
rf_clf.fit(X_train, y_train)
rf_y_pred = rf_clf.predict(X_test)
print("Best Parameters:", rf_clf.best_params_)
print("Classification Report:")
print(classification_report(y_test, rf_y_pred, target_names=le_mode.classes_))

# --- 3. Improved XGBoost Classifier ---
print("\n=== Gradient Boosting (XGBoost) ===")
xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}
xgb_clf = RandomizedSearchCV(
    XGBClassifier(eval_metric='mlogloss', random_state=42),
    xgb_params,
    n_iter=10,
    cv=3,
    random_state=42
)
xgb_clf.fit(X_train, y_train)
xgb_y_pred = xgb_clf.predict(X_test)
print("Best Parameters:", xgb_clf.best_params_)
print("Classification Report:")
print(classification_report(y_test, xgb_y_pred, target_names=le_mode.classes_))

# --- 4. Improved Neural Network (MLP) ---
print("\n=== Neural Network (MLP) ===")
mlp_params = {
    'hidden_layer_sizes': [(128, 64), (128, 64, 32), (256, 128)],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [400, 500]
}
mlp_clf = RandomizedSearchCV(
    MLPClassifier(random_state=42),
    mlp_params,
    n_iter=10,
    cv=3,
    random_state=42
)
mlp_clf.fit(X_train_scaled, y_train)
mlp_y_pred = mlp_clf.predict(X_test_scaled)
print("Best Parameters:", mlp_clf.best_params_)
print("Classification Report:")
print(classification_report(y_test, mlp_y_pred, target_names=le_mode.classes_))



""" import pandas as pd
import geopandas as gpd
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

# Load the GeoJSON dataset
file_path = 'filtered_file.geojson'
data = gpd.read_file(file_path)

# Preprocessing
# Drop unnecessary columns
if 'geometry' in data.columns:
    data = data.drop(columns=['geometry'])

# Translate and simplify French categories to English
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
data['mode'] = data['mode'].map(translations)

# Handle missing target values
data = data.dropna(subset=['mode'])

# Encode categorical variables
le_mode = LabelEncoder()
data['mode'] = le_mode.fit_transform(data['mode'])

if 'purpose' in data.columns:
    le_purpose = LabelEncoder()
    data['purpose'] = le_purpose.fit_transform(data['purpose'].fillna('Unknown'))

# Separate features and target
X = data.drop(columns=['mode'])
y = data['mode']

# Handle datetime columns
datetime_columns = X.select_dtypes(include=['datetime64']).columns
for col in datetime_columns:
    X[col] = pd.to_datetime(X[col])
    X[f'{col}_year'] = X[col].dt.year
    X[f'{col}_month'] = X[col].dt.month
    X[f'{col}_day'] = X[col].dt.day
    X[f'{col}_hour'] = X[col].dt.hour
    X.drop(columns=[col], inplace=True)

# Handle missing values in features
imputer = SimpleImputer(strategy='mean')  # Replace NaN with mean
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Address class imbalance with SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features for MLP and Deep Learning
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- PyTorch Deep Learning Model ---
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.long)  # Ensure labels are NumPy array

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# Prepare data for PyTorch
train_dataset = CustomDataset(X_train_scaled, y_train)
test_dataset = CustomDataset(X_test_scaled, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize PyTorch model
input_size = X_train.shape[1]
num_classes = len(np.unique(y))
model = NeuralNetwork(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Training loop
epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

# Evaluation
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

print("\n=== PyTorch Neural Network ===")
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=le_mode.classes_))

# --- 1. Improved Decision Tree Classifier ---
print("\n=== Decision Tree Classifier ===")
dt_params = {
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None]
}
dt_clf = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_params,
    n_iter=10,
    cv=3,
    random_state=42
)
dt_clf.fit(X_train, y_train)
dt_y_pred = dt_clf.predict(X_test)
print("Best Parameters:", dt_clf.best_params_)
print("Classification Report:")
print(classification_report(y_test, dt_y_pred, target_names=le_mode.classes_))

# --- 2. Improved Random Forest Classifier ---
print("\n=== Random Forest Classifier ===")
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced']
}
rf_clf = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    n_iter=10,
    cv=3,
    random_state=42
)
rf_clf.fit(X_train, y_train)
rf_y_pred = rf_clf.predict(X_test)
print("Best Parameters:", rf_clf.best_params_)
print("Classification Report:")
print(classification_report(y_test, rf_y_pred, target_names=le_mode.classes_))

# --- 3. Improved XGBoost Classifier ---
print("\n=== Gradient Boosting (XGBoost) ===")
xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}
xgb_clf = RandomizedSearchCV(
    XGBClassifier(eval_metric='mlogloss', random_state=42),
    xgb_params,
    n_iter=10,
    cv=3,
    random_state=42
)
xgb_clf.fit(X_train, y_train)
xgb_y_pred = xgb_clf.predict(X_test)
print("Best Parameters:", xgb_clf.best_params_)
print("Classification Report:")
print(classification_report(y_test, xgb_y_pred, target_names=le_mode.classes_))

# --- 4. Improved Neural Network (MLP) ---
print("\n=== Neural Network (MLP) ===")
mlp_params = {
    'hidden_layer_sizes': [(128, 64), (128, 64, 32), (256, 128)],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [400, 500]
}
mlp_clf = RandomizedSearchCV(
    MLPClassifier(random_state=42),
    mlp_params,
    n_iter=10,
    cv=3,
    random_state=42
)
mlp_clf.fit(X_train_scaled, y_train)
mlp_y_pred = mlp_clf.predict(X_test_scaled)
print("Best Parameters:", mlp_clf.best_params_)
print("Classification Report:")
print(classification_report(y_test, mlp_y_pred, target_names=le_mode.classes_))
 """






