# YOLOv8 Training Results Viewer with Dynamic Run Selection (No Plot)
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from PIL import Image, ImageTk

# Helper: Get the latest run folder
def get_latest_run(base_path="runs/train"):
    if not os.path.exists(base_path):
        return ""
    subdirs = [os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if not subdirs:
        return ""
    latest = max(subdirs, key=os.path.getmtime)
    return latest

# Load CSV
def load_metrics_csv(log_dir):
    csv_path = os.path.join(log_dir, "results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError("results.csv not found in the training output directory")
    return pd.read_csv(csv_path)

# Load PNG image
def load_model_summary_image(log_dir):
    img_path = os.path.join(log_dir, "results.png")
    if not os.path.exists(img_path):
        return None
    img = Image.open(img_path)
    img.thumbnail((500, 400))
    return ImageTk.PhotoImage(img)

# Load class-wise metrics

def load_val_summary_text(log_dir):
    txt_path = os.path.join(log_dir, "val_summary.txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            return f.read()
    return "⚠️ No class-wise evaluation summary found."

# Final row metrics from CSV
def extract_final_eval_summary(df):
    try:
        last = df.iloc[-1]
        summary = f"Final Evaluation Summary:\n"
        summary += f"mAP@0.5: {last.get('metrics/mAP50', 0):.3f}\n"
        summary += f"mAP@0.5:0.95: {last.get('metrics/mAP50-95', 0):.3f}\n"
        summary += f"Box Loss: {last.get('box_loss', 0):.3f} | Class Loss: {last.get('cls_loss', 0):.3f} | DFL Loss: {last.get('dfl_loss', 0):.3f}\n"
        return summary
    except Exception as e:
        return f"Error reading final results: {e}"

# GUI
class TrainingResultsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Training Results Viewer")
        self.root.geometry("820x750")

        self.dropdown = ttk.Combobox(root, postcommand=self.update_run_list, state="readonly")
        self.dropdown.pack(pady=10)
        self.dropdown.bind("<<ComboboxSelected>>", self.on_selection)

        self.summary_label = tk.Label(root)
        self.summary_label.pack()

        self.tree = ttk.Treeview(root, show='headings', height=8)
        self.tree.pack(expand=False, fill='x', padx=10, pady=5)

        self.summary_text = tk.Text(root, height=12, font=("Courier", 10))
        self.summary_text.pack(padx=10, pady=10, fill='x')

        self.load_run(get_latest_run())

    def update_run_list(self):
        base_path = "runs/train"
        dirs = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
        self.dropdown['values'] = dirs
        latest = get_latest_run(base_path)
        if latest:
            self.dropdown.set(os.path.basename(latest))

    def on_selection(self, event):
        run_name = self.dropdown.get()
        run_path = os.path.join("runs/train", run_name)
        self.load_run(run_path)

    def load_run(self, log_dir):
        try:
            self.df = load_metrics_csv(log_dir)
            self.class_summary = load_val_summary_text(log_dir)
            self.summary_img = load_model_summary_image(log_dir)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        for widget in self.summary_label.winfo_children():
            widget.destroy()

        if self.summary_img:
            self.summary_label.config(image=self.summary_img)
            self.summary_label.image = self.summary_img
        else:
            self.summary_label.config(text="No summary image found")

        self.tree.delete(*self.tree.get_children())
        self.tree['columns'] = list(self.df.columns)
        for col in self.df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=80)
        for _, row in self.df.iterrows():
            self.tree.insert("", "end", values=list(row))

        self.summary_text.config(state='normal')
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, extract_final_eval_summary(self.df))
        self.summary_text.insert(tk.END, "\n" + self.class_summary)
        self.summary_text.config(state='disabled')

if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingResultsGUI(root)
    root.mainloop()
