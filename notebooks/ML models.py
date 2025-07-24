import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
import matplotlib.pyplot as plt

# Step 1: Load dataset
df = pd.read_csv(r'C:\Users\Admin\Desktop\Intrusion-detection-project\Intrusion-Detection-Project\datastes\Radardata_updatedheat.csv')  # Replace with your actual file path

# Step 2: Feature selection
X = df[['speed', 'distance', 'angle', 'risk_range']]
y = df['label']

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Decision Tree model
clf = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# Step 5: Evaluation
y_pred = clf.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 6: Tree visualization
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, feature_names=X.columns, class_names=['Human', 'Vehicle', 'Animal'], filled=True)
plt.title("Decision Tree Visualization")
plt.show()
