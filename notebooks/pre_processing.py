# radar_preprocessing_with_labeling.py
# --------------------------------------------------
# Preprocessing radar data for object classification
# Includes cleaning, labeling, balancing, normalization

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Step 1: Load radar CSV data
input_file = r'C:\Users\Admin\Desktop\Intrusion-detection-project\Intrusion-Detection-Project\datastes\RadarData_processed.csv'  # Change to your file path
output_file = r'C:\Users\Admin\Desktop\Intrusion-detection-project\Intrusion-Detection-Project\datastes\RadarData_sampled.csv'


print("Loading data...")
df = pd.read_csv(input_file)

# Step 2: Clean missing and unrealistic values
df.dropna(inplace=True)
df = df[(df['x'].abs() < 50) & (df['y'].abs() < 50)]

'''# Step 3: Feature extraction
df['distance'] = np.sqrt(df['x']**2 + df['y']**2 + df.get('z', 0)**2)
df['angle'] = np.degrees(np.arctan2(df['y'], df['x']))
if 'doppler' in df.columns:
    df['speed'] = df['doppler']
elif 'range' in df.columns and 'time' in df.columns:
    df['speed'] = df['range'] / df['time']'''

# Step 4: Labeling the Data
# This example uses a rule-based labeling for demo.
# Replace with actual labels if available in your dataset.
# E.g., human: 0, vehicle: 1, animal: 2
df['label'] = df['distance'].apply(lambda d: 0 if d < 5 else (1 if d < 20 else 2))

#for intensity
df['intensity'] = 1 / (df['distance'] ** 4 + 1e-6)  # Add epsilon to avoid division by zero

# Step 5: Resampling / Balancing Classes
X = df[['x', 'y', 'velocity', 'intensity', 'distance', 'speed', 'angle']]
y = df['label']

print("Balancing dataset using SMOTE...")
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# Step 6: Normalize / Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)

# Step 7: Save Processed Data
final_df = pd.DataFrame(X_scaled, columns=X.columns)
final_df['label'] = y_res
final_df.to_csv(output_file, index=False)

print(f"Preprocessed and balanced data saved to {output_file}")

