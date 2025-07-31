import tkinter as tk
from tkinter import messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class RadarDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Radar Intrusion Detection Dashboard")
        self.root.geometry("1200x900")

        self.data = pd.read_csv("datastes\Radardata_updatedheat.csv")
        self.data = self.data.drop(columns=[col for col in self.data.columns if "Unnamed" in col])

        # Refresh Graphs button
        self.refresh_button = tk.Button(root, text="Refresh Graphs", command=self.plot_graphs, bg='lightgreen', font=('Arial', 12))
        self.refresh_button.pack(pady=10)

        # Frame for graphs
        self.graph_frame = tk.Frame(root)
        self.graph_frame.pack(fill="both", expand=True)

        self.plot_graphs()

    def plot_graphs(self):
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        fig, axes = plt.subplots(3, 2, figsize=(16, 12))

        # 1. Scatter plot (x vs y)
        scatter = axes[0, 0].scatter(self.data['x'], self.data['y'], c=self.data['label'], cmap='coolwarm', alpha=0.6)
        axes[0, 0].set_title("Position Scatter Plot (x vs y)")
        axes[0, 0].set_xlabel("x")
        axes[0, 0].set_ylabel("y")

        # 2. Histogram of risk_range
        self.data['risk_range'].plot(kind='hist', ax=axes[0, 1], bins=30, color='orange', alpha=0.7)
        axes[0, 1].set_title("Risk Range Distribution")
        axes[0, 1].set_xlabel("Risk Range")

        # 3. Bar plot of label counts
        label_counts = self.data['label'].value_counts().sort_index()
        label_counts.plot(kind='bar', ax=axes[1, 0], color='green', alpha=0.6)
        axes[1, 0].set_title("Intrusion Labels Count (Bar)")
        axes[1, 0].set_xlabel("Label")
        axes[1, 0].set_ylabel("Count")

        # 4. Pie chart of label counts
        label_counts.plot(kind='pie', ax=axes[1, 1], autopct='%1.1f%%', startangle=90, cmap='Set3')
        axes[1, 1].set_title("Intrusion Labels Distribution (Pie)")
        axes[1, 1].set_ylabel("")

        # 5. Triple Line Graph (x, y, angle)
        axes[2, 0].plot(self.data['x'][:300], label='x', linestyle='-', color='blue')
        axes[2, 0].plot(self.data['y'][:300], label='y', linestyle='--', color='red')
        axes[2, 0].plot(self.data['angle'][:300], label='angle', linestyle='-.', color='purple')
        axes[2, 0].set_title("Triple Line Plot: x, y, angle")
        axes[2, 0].legend()

        # 6. Distance vs Intensity (Scatter)
        axes[2, 1].scatter(self.data['distance'], self.data['intensity'], alpha=0.5, color='darkcyan')
        axes[2, 1].set_title("Distance vs Intensity")
        axes[2, 1].set_xlabel("Distance")
        axes[2, 1].set_ylabel("Intensity")

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

if __name__ == '__main__':
    root = tk.Tk()
    app = RadarDashboard(root)
    root.mainloop()
