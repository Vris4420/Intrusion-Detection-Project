# radar_feature_extraction.py
# This script reads radar point cloud data from a CSV file,
# extracts features such as distance, speed (Doppler), and angle
# and writes the enhanced dataset back to a new CSV file.
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Load radar CSV file
input_csv = r"C:\Users\Admin\Desktop\Intrusion-detection-project\Intrusion-Detection-Project\datastes\RadarData_raw.csv"
output_csv = r"C:\Users\Admin\Desktop\Intrusion-detection-project\Intrusion-Detection-Project\datastes\RadarData_processed.csv"

#ReadCSV
df=pd.read_csv(input_csv)

# Drop rows with missing values
df = df.dropna()

# Feature: Euclidean Distance from radar origin
df['distance'] = np.sqrt(df['x']**2 + df['y']**2 + df.get('z', 0)**2)

df['speed']=abs(df['velocity'])

# Optional Feature: Angle calculation (for 2D)
df['angle'] = np.degrees(np.arctan2(df['y'], df['x']))

#k-means clustering 
features = df[['x', 'y','z']]
kmeans = KMeans(n_clusters=3).fit(features)
df['cluster'] = kmeans.labels_

# Save new dataset
df.to_csv(output_csv, index=False)
print(f"Features added and saved to {output_csv}")
