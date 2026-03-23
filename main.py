import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
print("Loading dataset...")
df = pd.read_csv("titanic.csv")

# Show sample rows
print("\nFirst 5 rows:")
print(df.head())

# Show column names
print("\nColumns in dataset:")
print(df.columns)

# Select only useful columns
df = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'survived']]

# Fill missing values
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Convert text columns into numbers
sex_encoder = LabelEncoder()
embarked_encoder = LabelEncoder()

df['sex'] = sex_encoder.fit_transform(df['sex'])
df['embarked'] = embarked_encoder.fit_transform(df['embarked'])

# Features and target
X = df.drop('survived', axis=1)
y = df['survived']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Logistic Regression Model
# -----------------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
log_accuracy = accuracy_score(y_test, log_pred)

print("\n=== Logistic Regression Results ===")
print("Accuracy:", log_accuracy)
print("\nClassification Report:")
print(classification_report(y_test, log_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, log_pred))

# Feature importance for logistic regression
feature_names = X.columns
coefficients = log_model.coef_[0]

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': coefficients
}).sort_values(by='Importance', ascending=False)

print("\nLogistic Regression Feature Importance:")
print(importance_df)

plt.figure(figsize=(8, 5))
plt.bar(importance_df['Feature'], importance_df['Importance'])
plt.xticks(rotation=45)
plt.title("Feature Importance in Logistic Regression")
plt.tight_layout()
plt.show()

# -----------------------------
# Decision Tree Model
# -----------------------------
tree_model = DecisionTreeClassifier(random_state=42, max_depth=4)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)
tree_accuracy = accuracy_score(y_test, tree_pred)

print("\n=== Decision Tree Results ===")
print("Accuracy:", tree_accuracy)
print("\nClassification Report:")
print(classification_report(y_test, tree_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, tree_pred))

# Feature importance for decision tree
tree_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': tree_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nDecision Tree Feature Importance:")
print(tree_importance_df)

plt.figure(figsize=(8, 5))
plt.bar(tree_importance_df['Feature'], tree_importance_df['Importance'])
plt.xticks(rotation=45)
plt.title("Feature Importance in Decision Tree")
plt.tight_layout()
plt.show()

# -----------------------------
# Compare model accuracy
# -----------------------------
comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree'],
    'Accuracy': [log_accuracy, tree_accuracy]
})

print("\n=== Model Comparison ===")
print(comparison_df)

plt.figure(figsize=(6, 4))
plt.bar(comparison_df['Model'], comparison_df['Accuracy'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# -----------------------------
# graph: survival rate by sex
# -----------------------------
survival_by_sex = pd.read_csv("titanic.csv").groupby("sex")["survived"].mean()

plt.figure(figsize=(6, 4))
plt.bar(survival_by_sex.index, survival_by_sex.values)
plt.title("Survival Rate by Sex")
plt.ylabel("Average Survival Rate")
plt.tight_layout()
plt.show()