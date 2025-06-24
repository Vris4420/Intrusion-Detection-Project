
# Radar-Based Intrusion Detection Project

## Overview
This project focuses on detecting and classifying intrusions (human, animal, vehicle) using mmWave radar data and data analytics. It utilizes public datasets and machine learning to build a smart perimeter monitoring system suitable for defense and surveillance applications.

## Project Pipeline

### 1. Data Collection
- Source: Public mmWave radar datasets (e.g., RadarScenes, CARRADA, TI People Counting)
- Format: Point clouds or range-Doppler heatmaps
- Tools: Python, Pandas, NumPy

### 2. Data Preprocessing
- Clean and filter raw radar data
- Feature extraction: range, doppler, angle, intensity
- Labeling: Manual or based on metadata
- Balancing classes using oversampling/undersampling
- Normalization using StandardScaler

### 3. Machine Learning Model
- Models used: Random Forest, SVM, CNN (for heatmaps)
- Frameworks: Scikit-learn, TensorFlow
- Input: Extracted features (X, Y, Doppler, intensity)
- Output: Classified object (human, animal, vehicle)

### 4. Visualization Dashboard
- Real-time plotting with Plotly Dash
- Heatmaps of intrusion zones
- Historical logs and motion patterns

### 5. Deployment
- Platform: Local server or Jetson Nano/Raspberry Pi
- Includes: Live monitoring, alert generation, and data logging

## Dependencies
- Python 3.8+
- Libraries: pandas, numpy, matplotlib, scikit-learn, seaborn, plotly, dash, joblib

## Folder Structure
```
project/
├── data/                # Raw and processed datasets
├── src/                 # Python scripts (preprocessing, training, visualization)
├── models/              # Saved ML models
├── logs/                # Alert and detection logs
├── dashboard/           # Dash app for real-time display
└── README.md            # Project overview and guide
```

## How to Run
1. Clone the repository.
2. Place your dataset inside the `data/` folder.
3. Run `src/data_preprocessing.py` to clean and extract features.
4. Train your model using `src/train_model.py`.
5. Launch dashboard using `python dashboard/app.py`.

## Authors
- Your Name (ECE + Data Analytics Developer)

## License
This project is open-source and free to use for academic and non-commercial use.
