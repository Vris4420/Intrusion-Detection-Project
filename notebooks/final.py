import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
import matplotlib.pyplot as plt
import pickle
from joblib import load,dump
import torch
import torch.nn as nn


# Uncomment the line below and comment the synthetic data generation above when using your CSV
df = pd.read_csv(r'C:\Users\Admin\Desktop\Intrusion-detection-project\Intrusion-Detection-Project\datastes\RadarData_updatedheat.csv')

print("Dataset shape:", df.shape)
print("Label distribution:")
print(df['label'].value_counts())

# Step 2: Feature selection
X = df[['speed', 'distance', 'angle', 'risk_range']]
y = df['label']

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Step 4: Decision Tree model
print("\n" + "="*50)
print("Training Decision Tree Classifier")
print("="*50)

clf = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# Evaluation
y_pred = clf.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred)
print("Decision Tree Accuracy:", dt_accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 5: Random Forest model
print("\n" + "="*50)
print("Training Random Forest Classifier")
print("="*50)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluation
y_pred_rf = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", rf_accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# Step 6: Save models in different formats

# Method: Save as .pt files (PyTorch format)
# For scikit-learn models, we'll wrap them in a PyTorch-compatible format

class SklearnModelWrapper:
    def __init__(self, sklearn_model, feature_names=None, class_names=None):
        self.sklearn_model = sklearn_model
        self.feature_names = feature_names
        self.class_names = class_names
        self.model_type = type(sklearn_model).__name__

    def predict(self, X):
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        return self.sklearn_model.predict(X)
    
    def predict_proba(self, X):
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        return self.sklearn_model.predict_proba(X)

# Wrap the models
dt_wrapper = SklearnModelWrapper( clf, 
    feature_names=list(X.columns), 
    class_names=list(clf.classes_)
)

rf_wrapper = SklearnModelWrapper( clf, 
    feature_names=list(X.columns), 
    class_names=list(clf.classes_)
)

# Save as .pt files
dump(dt_wrapper, r"C:\Users\Admin\Desktop\Intrusion-detection-project\Intrusion-Detection-Project\models\decision_tree_model.pt")
print('''Decision Tree model saved as

'decision_tree_model.pt''')
dump(dt_wrapper, r"C:\Users\Admin\Desktop\Intrusion-detection-project\Intrusion-Detection-Project\models\random_forest_model.pt")
print("Random Forest model saved as 'random_forest_model.pt'")

# Step 7: Save model metadata
model_info = {
    'decision_tree': {
        'accuracy': dt_accuracy,
        'max_depth': clf.max_depth,
        'criterion': clf.criterion,
        'feature_names': list(X.columns),
        'class_names': list(clf.classes_),
        'n_features': X.shape[1],
        'n_classes': len(clf.classes_)
    },
    'random_forest': {
        'accuracy': rf_accuracy,
        'n_estimators': rf.n_estimators,
        'feature_names': list(X.columns),
        'class_names': list(rf.classes_),
        'n_features': X.shape[1],
        'n_classes': len(rf.classes_)
    }
}

dump(dt_wrapper, r"C:\Users\Admin\Desktop\Intrusion-detection-project\Intrusion-Detection-Project\models\model_metadata.pt")
print("Model metadata saved as 'model_metadata.pt'")

# Step 8: Demonstrate loading the .pt files
print("\n" + "="*50)
print("Testing Model Loading")
print("="*50)

# Load the models
loaded_dt = load(r"C:\Users\Admin\Desktop\Intrusion-detection-project\Intrusion-Detection-Project\models\decision_tree_model.pt")
loaded_rf = load(r"C:\Users\Admin\Desktop\Intrusion-detection-project\Intrusion-Detection-Project\models\random_forest_model.pt ")
loaded_metadata =load(r"C:\Users\Admin\Desktop\Intrusion-detection-project\Intrusion-Detection-Project\models\model_metadata.pt  ")
# Load the full model temporarily'''

# Save only the weights now
torch.save(loaded_dt,r'C:\Users\Admin\Desktop\Intrusion-detection-project\Intrusion-Detection-Project\models\decision_tree_model.pt')

print("Successfully loaded all models!")
print(f"Decision Tree type: {loaded_dt.model_type}")
print(f"Random Forest type: {loaded_rf.model_type}")

# Test predictions with loaded models
sample_data = X_test.iloc[:5].values
dt_predictions = loaded_dt.predict(sample_data)
rf_predictions = loaded_rf.predict(sample_data)

print("\nSample predictions:")
print("Features (first 5 test samples):")
print(X_test.iloc[:5].values)
print(f"Decision Tree predictions: {dt_predictions}")
print(f"Random Forest predictions: {rf_predictions}")
print(f"Actual labels: {y_test.iloc[:5].values}")

# Step 9: Visualize Decision Tree (optional)
try:
    plt.figure(figsize=(15, 10))
    tree.plot_tree(clf, 
                   feature_names=X.columns, 
                   class_names=[str(cls) for cls in clf.classes_], 
                   filled=True, 
                   rounded=True,
                   fontsize=10)
    plt.title("Decision Tree Visualization")
    plt.tight_layout()
    plt.savefig('decision_tree_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Decision tree visualization saved as 'decision_tree_visualization.png'")
except Exception as e:
    print(f"Could not create visualization: {e}")

print("\n" + "="*50)
print("Summary")
print("="*50)
print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print("\nFiles created:")
print("- decision_tree_model.pt")
print("- random_forest_model.pt") 
print("- model_metadata.pt")
print("- decision_tree_model.pkl")
print("- random_forest_model.pkl")
print("- decision_tree_visualization.png (if matplotlib works)")
