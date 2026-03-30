import pandas as pd
# Load the dataset
df = pd.read_csv('oral_cancer_prediction.csv')

# Display the first 5 rows
display(df.head())

# Print concise summary
df.info()

# Display descriptive statistics
display(df.describe())

# Handle missing values
# For numerical columns, fill with the mean
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

# For categorical columns, fill with the mode
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# Verify that there are no more missing values
df.info()

# Encode categorical variables using one-hot encoding
# Exclude the target variable 'Oral Cancer (Diagnosis)'
categorical_cols_to_encode = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols_to_encode.remove('Oral Cancer (Diagnosis)')

df_encoded = pd.get_dummies(df, columns=categorical_cols_to_encode, drop_first=True)

# Display the first 5 rows of the encoded dataframe
display(df_encoded.head())

# Print concise summary of the encoded dataframe
df_encoded.info()

from sklearn.preprocessing import StandardScaler

# Select numerical columns to scale (excluding the target and ID)
numerical_cols_to_scale = df_encoded.select_dtypes(include=['float64', 'int64']).columns.tolist()
numerical_cols_to_scale.remove('ID') # Assuming 'ID' is not a feature
# 'Oral Cancer (Diagnosis)' is the target variable and should not be scaled

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply StandardScaler to the numerical columns
df_encoded[numerical_cols_to_scale] = scaler.fit_transform(df_encoded[numerical_cols_to_scale])

# Display the first 5 rows of the scaled dataframe
display(df_encoded.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Separate features (X) and target variable (y)
X = df_encoded.drop('Oral Cancer (Diagnosis)', axis=1)
y = df_encoded['Oral Cancer (Diagnosis)']

# Convert the target variable to numerical if it's not already
y = y.map({'No': 0, 'Yes': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Initialize Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)

# Initialize Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Train the Decision Tree model
dt_model.fit(X_train, y_train)

# Train the Random Forest model
rf_model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score

# Evaluate Logistic Regression Model
y_pred_lr = model.predict(X_test)
y_pred_proba_lr = model.predict_proba(X_test)[:, 1]

accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_pred_proba_lr)

print("--- Logistic Regression Performance ---")
print(f"Accuracy: {accuracy_lr:.4f}")
print(f"Precision: {precision_lr:.4f}")
print(f"Recall: {recall_lr:.4f}")
print(f"F1-Score: {f1_lr:.4f}")
print(f"AUC-ROC: {auc_lr:.4f}")
print("-" * 35)

# Evaluate Decision Tree Model
y_pred_dt = dt_model.predict(X_test)
y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]

accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
auc_dt = roc_auc_score(y_test, y_pred_proba_dt)

print("--- Decision Tree Performance ---")
print(f"Accuracy: {accuracy_dt:.4f}")
print(f"Precision: {precision_dt:.4f}")
print(f"Recall: {recall_dt:.4f}")
print(f"F1-Score: {f1_dt:.4f}")
print(f"AUC-ROC: {auc_dt:.4f}")
print("-" * 35)


# Evaluate Random Forest Model
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

print("--- Random Forest Performance ---")
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1-Score: {f1_rf:.4f}")
print(f"AUC-ROC: {auc_rf:.4f}")
print("-" * 35)

import pandas as pd

# Create a dictionary to hold the metrics
metrics_data = {
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
    'Accuracy': [accuracy_lr, accuracy_dt, accuracy_rf],
    'Precision': [precision_lr, precision_dt, precision_rf],
    'Recall': [recall_lr, recall_dt, recall_rf],
    'F1-Score': [f1_lr, f1_dt, f1_rf],
    'AUC-ROC': [auc_lr, auc_dt, auc_rf]
}

# Create a pandas DataFrame from the dictionary
metrics_df = pd.DataFrame(metrics_data)

# Display the DataFrame
display(metrics_df)

from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier

# Re-initialize models with increased max_iter for Logistic Regression
model_cv = LogisticRegression(max_iter=1000) # Increased iterations
dt_model_cv = DecisionTreeClassifier(random_state=42)
rf_model_cv = RandomForestClassifier(random_state=42)

# Perform cross-validation for Logistic Regression
cv_scores_lr = cross_val_score(model_cv, X, y, cv=5, scoring='accuracy') # Using 5-fold cross-validation
print("--- Logistic Regression Cross-Validation (Accuracy) ---")
print(f"Mean Accuracy: {cv_scores_lr.mean():.4f}")
print(f"Standard Deviation: {cv_scores_lr.std():.4f}")
print("-" * 50)

# Perform cross-validation for Decision Tree
cv_scores_dt = cross_val_score(dt_model_cv, X, y, cv=5, scoring='accuracy') # Using 5-fold cross-validation
print("--- Decision Tree Cross-Validation (Accuracy) ---")
print(f"Mean Accuracy: {cv_scores_dt.mean():.4f}")
print(f"Standard Deviation: {cv_scores_dt.std():.4f}")
print("-" * 50)

# Perform cross-validation for Random Forest
cv_scores_rf = cross_val_score(rf_model_cv, X, y, cv=5, scoring='accuracy') # Using 5-fold cross-validation
print("--- Random Forest Cross-Validation (Accuracy) ---")
print(f"Mean Accuracy: {cv_scores_rf.mean():.4f}")
print(f"Standard Deviation: {cv_scores_rf.std():.4f}")
print("-" * 50)

import seaborn as sns
import matplotlib.pyplot as plt

# Visualize the distribution of the target variable
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Oral Cancer (Diagnosis)')
plt.title('Distribution of Oral Cancer Cases')
plt.xlabel('Oral Cancer Diagnosis')
plt.ylabel('Count')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Select numerical columns from the original dataframe (df)
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
# Exclude 'ID' if it's just an identifier
if 'ID' in numerical_cols:
    numerical_cols.remove('ID')

# Create histograms for numerical features
df[numerical_cols].hist(bins=30, figsize=(15, 10))
plt.suptitle('Distributions of Numerical Features', y=1.02, ha='center', fontsize=16)
plt.tight_layout()
plt.show()

# Visualize the distributions of categorical features
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
# Exclude the target variable as its distribution is already visualized
if 'Oral Cancer (Diagnosis)' in categorical_cols:
    categorical_cols.remove('Oral Cancer (Diagnosis)')

for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y=col, order=df[col].value_counts().index, palette='viridis', hue=col, legend=False)
    plt.title(f'Distribution of {col}')
    plt.xlabel('Count')
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

from sklearn.model_selection import cross_val_score

# Perform cross-validation for Logistic Regression
cv_scores_lr = cross_val_score(model_cv, X, y, cv=5, scoring='accuracy') # Using 5-fold cross-validation
print("--- Logistic Regression Cross-Validation (Accuracy) ---")
print(f"Mean Accuracy: {cv_scores_lr.mean():.4f}")
print(f"Standard Deviation: {cv_scores_lr.std():.4f}")
print("-" * 50)

# Perform cross-validation for Decision Tree
cv_scores_dt = cross_val_score(dt_model_cv, X, y, cv=5, scoring='accuracy') # Using 5-fold cross-validation
print("--- Decision Tree Cross-Validation (Accuracy) ---")
print(f"Mean Accuracy: {cv_scores_dt.mean():.4f}")
print(f"Standard Deviation: {cv_scores_dt.std():.4f}")
print("-" * 50)

# Perform cross-validation for Random Forest
cv_scores_rf = cross_val_score(rf_model_cv, X, y, cv=5, scoring='accuracy') # Using 5-fold cross-validation
print("--- Random Forest Cross-Validation (Accuracy) ---")
print(f"Mean Accuracy: {cv_scores_rf.mean():.4f}")
print(f"Standard Deviation: {cv_scores_rf.std():.4f}")
print("-" * 50)

import pandas as pd

# Create a dictionary to hold the metrics
metrics_data = {
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
    'Accuracy': [accuracy_lr, accuracy_dt, accuracy_rf],
    'Precision': [precision_lr, precision_dt, precision_rf],
    'Recall': [recall_lr, recall_dt, recall_rf],
    'F1-Score': [f1_lr, f1_dt, f1_rf],
    'AUC-ROC': [auc_lr, auc_dt, auc_rf]
}

# Create a pandas DataFrame from the dictionary
metrics_df = pd.DataFrame(metrics_data)

# Display the DataFrame
display(metrics_df)

import pandas as pd

# Create a dictionary to hold the metrics
metrics_data = {
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
    'Accuracy': [accuracy_lr, accuracy_dt, accuracy_rf]
}

# Create a pandas DataFrame from the dictionary
accuracy_df = pd.DataFrame(metrics_data)

# Display the DataFrame
display(accuracy_df)

# Visualize the distributions of categorical features
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
# Exclude the target variable as its distribution is already visualized
if 'Oral Cancer (Diagnosis)' in categorical_cols:
    categorical_cols.remove('Oral Cancer (Diagnosis)')

for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y=col, order=df[col].value_counts().index, palette='viridis', hue=col, legend=False)
    plt.title(f'Distribution of {col}')
    plt.xlabel('Count')
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Select numerical columns from the original dataframe (df)
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
# Exclude 'ID' if it's just an identifier
if 'ID' in numerical_cols:
    numerical_cols.remove('ID')

# Create histograms for numerical features
df[numerical_cols].hist(bins=30, figsize=(15, 10))
plt.suptitle('Distributions of Numerical Features', y=1.02, ha='center', fontsize=16)
plt.tight_layout()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Visualize the distribution of the target variable
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Oral Cancer (Diagnosis)')
plt.title('Distribution of Oral Cancer Cases')
plt.xlabel('Oral Cancer Diagnosis')
plt.ylabel('Count')
plt.show()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Re-initialize and train the model before making predictions
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Display the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Oral Cancer', 'Oral Cancer'], yticklabels=['No Oral Cancer', 'Oral Cancer'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Separate features (X) and target variable (y)
X = df_encoded.drop('Oral Cancer (Diagnosis)', axis=1)
y = df_encoded['Oral Cancer (Diagnosis)']

# Convert the target variable to numerical if it's not already
y = y.map({'No': 0, 'Yes': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

from sklearn.preprocessing import StandardScaler

# Select numerical columns to scale (excluding the target and ID)
numerical_cols_to_scale = df_encoded.select_dtypes(include=['float64', 'int64']).columns.tolist()
numerical_cols_to_scale.remove('ID') # Assuming 'ID' is not a feature
# 'Oral Cancer (Diagnosis)' is the target variable and should not be scaled

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply StandardScaler to the numerical columns
df_encoded[numerical_cols_to_scale] = scaler.fit_transform(df_encoded[numerical_cols_to_scale])

# Display the first 5 rows of the scaled dataframe
display(df_encoded.head())

# Encode categorical variables using one-hot encoding
# Exclude the target variable 'Oral Cancer (Diagnosis)'
categorical_cols_to_encode = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols_to_encode.remove('Oral Cancer (Diagnosis)')

df_encoded = pd.get_dummies(df, columns=categorical_cols_to_encode, drop_first=True)

# Display the first 5 rows of the encoded dataframe
display(df_encoded.head())

# Print concise summary of the encoded dataframe
df_encoded.info()

# Handle missing values
# For numerical columns, fill with the mean
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

# For categorical columns, fill with the mode
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# Verify that there are no more missing values
df.info()

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Initialize Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)

# Initialize Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Train the Decision Tree model
dt_model.fit(X_train, y_train)

# Train the Random Forest model
rf_model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score

# Evaluate Logistic Regression Model
y_pred_lr = model.predict(X_test)
y_pred_proba_lr = model.predict_proba(X_test)[:, 1]

accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_pred_proba_lr)

print("--- Logistic Regression Performance ---")
print(f"Accuracy: {accuracy_lr:.4f}")
print(f"Precision: {precision_lr:.4f}")
print(f"Recall: {recall_lr:.4f}")
print(f"F1-Score: {f1_lr:.4f}")
print(f"AUC-ROC: {auc_lr:.4f}")
print("-" * 35)

# Evaluate Decision Tree Model
y_pred_dt = dt_model.predict(X_test)
y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]

accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
auc_dt = roc_auc_score(y_test, y_pred_proba_dt)

print("--- Decision Tree Performance ---")
print(f"Accuracy: {accuracy_dt:.4f}")
print(f"Precision: {precision_dt:.4f}")
print(f"Recall: {recall_dt:.4f}")
print(f"F1-Score: {f1_dt:.4f}")
print(f"AUC-ROC: {auc_dt:.4f}")
print("-" * 35)


# Evaluate Random Forest Model
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

print("--- Random Forest Performance ---")
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1-Score: {f1_rf:.4f}")
print(f"AUC-ROC: {auc_rf:.4f}")
print("-" * 35)

import pandas as pd

# Create a dictionary to hold the metrics
metrics_data = {
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
    'Accuracy': [accuracy_lr, accuracy_dt, accuracy_rf],
    'Precision': [precision_lr, precision_dt, precision_rf],
    'Recall': [recall_lr, recall_dt, recall_rf],
    'F1-Score': [f1_lr, f1_dt, f1_rf],
    'AUC-ROC': [auc_lr, auc_dt, auc_rf]
}

# Create a pandas DataFrame from the dictionary
metrics_df = pd.DataFrame(metrics_data)

# Display the DataFrame
display(metrics_df)