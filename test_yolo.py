import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import torch
import os

# === Globals ===
model = None
model_dropdown = None
panel = None
info_label = None
selected_image_path = None

# === List model paths ===
def list_model_paths(base_path="runs/train"):
    options = []
    for d in os.listdir(base_path):
        folder = os.path.join(base_path, d, "weights", "best.pt")
        if os.path.exists(folder):
            options.append(folder)
    return sorted(options)

# === Load YOLOv8 model ===
def load_model(model_path):
    global model
    try:
        model = YOLO(model_path)
        messagebox.showinfo("Model Loaded", f"Loaded model: {os.path.basename(model_path)}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model:\n{e}")

# === When model selected ===
def on_model_select(event):
    selected = model_dropdown.get()
    if selected:
        load_model(selected)

# === Detection button press ===
def run_detection():
    global selected_image_path, model
    if not model:
        messagebox.showwarning("Model Missing", "Please select a YOLO model first.")
        return
    if not selected_image_path:
        messagebox.showwarning("Image Missing", "Please upload an image first.")
        return

    results = model(selected_image_path, device="cuda" if torch.cuda.is_available() else "cpu", conf=0.25)[0]
    img = results.plot()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil.thumbnail((640, 480))
    img_tk = ImageTk.PhotoImage(img_pil)
    panel.config(image=img_tk)
    panel.image = img_tk
    panel.pack()

    try:
        pred_classes = [model.names[c] for c in results.boxes.cls.tolist()]
        confs = results.boxes.conf.tolist()
        if confs:
            avg_conf = sum(confs) / len(confs)
            info_text = f"Detected: {len(pred_classes)} object(s)\nClasses: {set(pred_classes)}\nAvg Confidence: {avg_conf:.2f}"
        else:
            info_text = "No objects detected."
    except Exception as e:
        info_text = f"Detection error: {e}"

    info_label.config(text=info_text)

# === Image select ===
def browse_image():
    global selected_image_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        selected_image_path = file_path
        info_label.config(text=f"Selected Image: {os.path.basename(file_path)}")

# === GUI Setup ===
root = tk.Tk()
root.title("YOLOv8 Model Selector & Image Tester")
root.geometry("720x700")

# Dropdown to select model
tk.Label(root, text="Select a trained YOLOv8 model:", font=("Arial", 12)).pack(pady=5)
model_dropdown = ttk.Combobox(root, values=list_model_paths(), width=80, state="readonly")
model_dropdown.bind("<<ComboboxSelected>>", on_model_select)
model_dropdown.pack(pady=5)

# Image select
tk.Button(root, text="📁 Upload Image", command=browse_image, font=("Arial", 14)).pack(pady=5)

# Detection trigger
tk.Button(root, text="🚀 Run Detection", command=run_detection, font=("Arial", 14)).pack(pady=10)

# Image display
panel = tk.Label(root)
panel.pack()

# Info label
info_label = tk.Label(root, text="", font=("Arial", 12), justify="left")
info_label.pack(pady=10)

root.mainloop()