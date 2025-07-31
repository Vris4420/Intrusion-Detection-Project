import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import subprocess
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog



# Load processed dataset (same every time)


CSV_PATH = r'datastes/Radardata_updatedheat.csv'
def select_and_process_input():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        return  # User cancelled

    output_path = "Radardata_updatedheat.csv"

    popup = tk.Toplevel(root)
    popup.title("Dataset Processing Status")
    popup.geometry("500x300")
    popup.configure(bg="#f0f0f0")

    status_label = tk.Label(popup, text="", font=("Helvetica", 12), bg="#f0f0f0", wraplength=480, justify="left")
    status_label.pack(pady=20)

    def close_popup():
        popup.destroy()

    close_button = tk.Button(popup, text="Close", command=close_popup, font=("Helvetica", 11),
                             bg="#d9534f", fg="white", padx=10, pady=5)
    close_button.pack(pady=20)

    if os.path.exists(output_path):
        status_label.config(text=f"✅ Processed file already exists:\n'{output_path}'\n\nLoading existing data.")
        return

    try:
        df = pd.read_csv(file_path)

        # --- Replace this with your real EDA/processing logic ---
        df = df.dropna()
        df['x'] = (df['x'] - df['x'].mean()) / df['x'].std()
        df['y'] = (df['y'] - df['y'].mean()) / df['y'].std()
        df['angle'] = (df['angle'] - df['angle'].min()) / (df['angle'].max() - df['angle'].min())
        # --------------------------------------------------------

        df.to_csv(output_path, index=False)

        status_label.config(text=f"✅ File processed successfully and saved as:\n'{output_path}'")

    except Exception as e:
        status_label.config(text=f"❌ Failed to process the file:\n{str(e)}", fg="red")



# Main app class
class IntrusionDetectionApp:
    def show_plot_in_popup(self,fig, title="Plot"):
        popup = tk.Toplevel()
        popup.title(title)
        popup.geometry("800x600")
        
        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        close_btn = tk.Button(popup, text="Close", command=popup.destroy, bg="#e53935", fg="white")
        close_btn.pack(pady=10)
    def __init__(self, root):
        self.root = root
        self.root.title("Intrusion Detection System Dashboard")
        self.root.geometry("1280x800")
        self.root.resizable(False, False)

        # Load and set background image
        bg_image_path = "images(2).jpg"
        try:
            bg_image = Image.open(bg_image_path)
            bg_image = bg_image.resize((1280, 800), Image.Resampling.LANCZOS)
            self.bg_photo = ImageTk.PhotoImage(bg_image)
            self.canvas = tk.Canvas(self.root, width=1280, height=800, highlightthickness=0)
            self.canvas.pack(fill="both", expand=True)

            self.bg_photo = ImageTk.PhotoImage(bg_image)
            self.canvas.create_image(0, 0, anchor="nw", image=self.bg_photo)
        except:
            print("Background image missing or corrupt.")

        self.setup_ui()

    def setup_ui(self):
        # EVERYTHING GOES INSIDE HERE ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

        def on_enter_green(e): e.widget.config(bg="#3e8e41")
        def on_leave_green(e): e.widget.config(bg="#4CAF50")
        def on_enter_blue(e): e.widget.config(bg="#3949ab")
        def on_leave_blue(e): e.widget.config(bg="#5c6bc0")

        def make_button(text, x, y, bg_color, hover_in, hover_out, command):
            btn = tk.Button(
                self.root, text=text, font=("Segoe UI", 13),
                bg=bg_color, fg="white", padx=25, pady=12, bd=0,
                activebackground=bg_color, command=command
            )
            btn.bind("<Enter>", hover_in)
            btn.bind("<Leave>", hover_out)
            self.canvas.create_window(x, y, window=btn)

        # Correct usage of self.<method> inside setup_ui
        make_button("Select & Process Dataset", 640, 100, "#4CAF50", on_enter_green, on_leave_green, select_and_process_input)
        make_button("Export Dataset", 440, 200, "#4CAF50", on_enter_green, on_leave_green, self.export_dataset)
        make_button("Run EDA", 840, 200, "#4CAF50", on_enter_green, on_leave_green, self.run_eda)
        make_button("Machine Learning Window", 640, 310, "#4CAF50", on_enter_green, on_leave_green, self.run_script_window)
        make_button("Open Dashboard", 640, 400, "#4CAF50", on_enter_green, on_leave_green, self.open_dashboard)
        make_button("Show Normalized Graph", 440, 500, "#5c6bc0", on_enter_blue, on_leave_blue, self.show_normalized_graph)
        make_button("Show Risk Radar Pie", 840, 500, "#5c6bc0", on_enter_blue, on_leave_blue, self.show_radar_pie_chart)

        # Arrows
        arrow_font = ("Segoe UI", 18)
        self.canvas.create_text(640, 150, text="↓", font=arrow_font, fill="white")
        self.canvas.create_text(640, 260, text="↓", font=arrow_font, fill="white")
        self.canvas.create_text(640, 360, text="↓", font=arrow_font, fill="white")
        self.canvas.create_text(640, 450, text="↓", font=arrow_font, fill="white")
    def export_dataset(self):
        try:
            df = pd.read_csv(CSV_PATH)
            export_path = "exported_dataset.csv"
            df.to_csv(export_path, index=False)
            messagebox.showinfo("Success", f"Dataset exported to {export_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    def run_eda(self):
        try:
            df = pd.read_csv(CSV_PATH)
            # Tabular Info
            info = StringIO()
            df.info(buf=info)
            head = df.head().to_string()

            # Show summary and head in pop-up
            messagebox.showinfo("Data Summary", f"{info.getvalue()}\n\n{head}")

            # Graphical EDA
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            df['label'].value_counts().plot(kind='pie', ax=axs[0, 0], autopct='%1.1f%%', title='Label Distribution')
            df[['x', 'y', 'angle']].plot(ax=axs[0, 1], title='X, Y, Angle Trends')
            df['distance'].plot(kind='hist', ax=axs[1, 0], bins=20, color='orange', title='Distance Histogram')
            df[['x', 'y']].plot(kind='scatter', x='x', y='y', ax=axs[1, 1], title='X vs Y Scatter')
            plt.tight_layout()
            plt.show()

        except Exception as e:
            messagebox.showerror("EDA Error", str(e))

    def run_script_window(self):
        try:
            win = tk.Toplevel(self.root)
            win.title("machine learning window")
            win.geometry("800x400")

            text_area = tk.Text(win, wrap='word', font=("Courier", 10))
            text_area.pack(expand=True, fill='both')

            result = subprocess.run(["python", "notebooks/final.py"], capture_output=True, text=True)
            output = result.stdout + "\n" + result.stderr
            text_area.insert(tk.END, output)
        except Exception as e:
            messagebox.showerror("Execution Failed", str(e))
    def show_normalized_graph(self):
        df = pd.read_csv(CSV_PATH)
        normalized_df = df.copy()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(normalized_df['x'], label='Normalized X', color='blue')
        ax.plot(normalized_df['y'], label='Normalized Y', color='green')
        ax.plot(normalized_df['angle'], label='Normalized Angle', color='red')
        ax.set_title('Normalized Data Trends')
        ax.set_xlabel('Index')
        ax.set_ylabel('Normalized Values')
        ax.legend()
        plt.tight_layout()

        self.show_plot_in_popup(fig, "Normalized Graph")
    
    def show_radar_pie_chart(self):
        df = pd.read_csv(CSV_PATH)
        # Calculate magnitude or use X and Y directly
        df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2)
        normalized_df = df.copy()

        high_risk = df[df['magnitude'] > df['magnitude'].quantile(0.66)]
        moderate_risk = df[(df['magnitude'] <= df['magnitude'].quantile(0.66)) & (df['magnitude'] > df['magnitude'].quantile(0.33))]
        safe = df[df['magnitude'] <= df['magnitude'].quantile(0.33)]

        sizes = [len(high_risk), len(moderate_risk), len(safe)]
        labels = ['High Risk', 'Moderate Risk', 'Safe']
        colors = ['red', 'yellow', 'green']

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Radar Zone Risk Pie Chart')
        plt.tight_layout()

        self.show_plot_in_popup(fig, "Radar Risk Zone Pie Chart")

        

    def open_dashboard(self):
        try:
            subprocess.run(["python", "dashboard.py"])
        except Exception as e:
            messagebox.showerror("Dashboard Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = IntrusionDetectionApp(root)
    root.mainloop()
