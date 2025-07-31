import pandas as pd
import requests
import json
import time

# 🔁 Replace this with your actual push URL
power_bi_url = "https://api.powerbi.com/beta/1b4eaaad-b887-48cf-8407-e99420eda2fb/datasets/36d0ded9-16a3-42a1-9445-6add4db61f8b/rows?experience=power-bi&key=uVnj%2Fo43Oa2QgWRGJ3r%2Bs906gSqT5IXLD9ymaAMo4%2F6t0z3D9EQqo%2BOiDzHvWjRa2XyZ6pS%2FdUsU2gMLfBu08A%3D%3D"

# 🔁 Replace with your actual CSV path
csv_path = r"C:\Users\Admin\Desktop\Intrusion-detection-project\Intrusion-Detection-Project\datastes\Radardata_updatedheat.csv"
last_index = -1

while True:
    df = pd.read_csv(csv_path)

    # Send only new data rows
    new_rows = df.iloc[last_index + 1:]
    if new_rows.empty:
        time.sleep(5)
        continue

    for _, row in new_rows.iterrows():
        data = [{
            "x": row["x"],
            "y": row["y"],
            "velocity": row["velocity"],
            "intensity": row["intensity"],
            "distance": row["distance"],
            "speed": row["speed"],
            "angle": row["angle"],
            "label": int(row["label"]),
            "risk_range": row["risk_range"]
        }]

        res = requests.post(power_bi_url, headers={"Content-Type": "application/json"},
                            data=json.dumps(data))
        print("Sent:", data, "| Status:", res.status_code)

    last_index = df.index[-1]
    time.sleep(5)
