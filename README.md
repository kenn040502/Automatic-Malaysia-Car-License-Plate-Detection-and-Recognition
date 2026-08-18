Automatic Malaysia Car License Plate Detection and Recognition (ALPR)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)

An AI-powered Automatic License Plate Recognition (ALPR) pipeline specifically designed and fine-tuned for Malaysian vehicle license plates. The system combines deep learning-based object detection to locate license plates in static images and real-time video streams with sequence-based character recognition to extract text accurately[cite: 1, 2].

Developed as part of **COS30018 Intelligent Systems** at Swinburne University of Technology[cite: 1, 2].

---

## Team Members (Group 2)

* **Kenneth Hui Hong CHUA** (102782494) — *YOLOv8 Model Lead & Optimization*
* **Andrew Kian Fui VOON** (102782821) — *SSD Model Development & Data Pipeline*[cite: 2]
* **Daniel Meng En PANG** (102776510) — *System Integration, GUI Development & Model Compilation*[cite: 2]
* **Jia Yi LIM** (104404062 / 102780207) — *CRNN Training, ViT Debugging & Data Annotation*[cite: 2]
* **Jewel Ze Syn LAI** (104404062) — *TrOCR (ViT) Model Lead & Tesseract Benchmarking*[cite: 2]

---

## System Architecture & Pipeline

The pipeline processes input through two primary deep learning stages[cite: 2]:

+------------------+      +--------------------------+      +-----------------------------+
| Input Media      | ---> | Object Detection Module  | ---> | Character Recognition (OCR) |
| (Images / Video) |      | (YOLOv8 / SSD)           |      | (CRNN / ViT / Tesseract)    |
+------------------+      +--------------------------+      +-----------------------------+
|                                    |
v                                    v
Crop Bounding Box                    Extracted Text Output


1. **Object Detection (Plate Localization):** Evaluates input frames to detect vehicles and pinpoint exact license plate coordinates using bounding boxes[cite: 2].
2. **Character Recognition (OCR):** Crops the detected region of interest (ROI) and decodes alphanumeric sequences[cite: 2].

---

## Performance Summary & Model Comparison

### 1. Object Detection Evaluation

Models were evaluated on a custom annotated dataset split into **80% Training**, **10% Validation**, and **10% Testing**[cite: 2].

| Model | mAP@0.5 | Precision | Recall | F1-Score | IoU Range |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8s** *(Selected)* | **0.979**[cite: 2] | **0.969**[cite: 2] | **0.990**[cite: 2] | **~0.980**[cite: 2] | **0.645 – 0.689**[cite: 2] |
| **SSD (VGG-16)** | 0.914[cite: 2] | 0.630[cite: 2] | 0.750[cite: 2] | 0.690[cite: 2] | 0.000 – 0.991[cite: 2] |

* **Key Takeaway:** YOLOv8 drastically outperformed SSD in stability, precision, and handling small bounding boxes across diverse lighting conditions[cite: 2].

---

### 2. Character Recognition (OCR) Evaluation

OCR architectures were evaluated using exact match word accuracy and Character Error Rate (CER) on a split of **70% Training**, **15% Validation**, and **15% Testing**[cite: 2].

| Technique | Character Accuracy | Word Accuracy | CER (Character Error Rate) | WER (Word Error Rate) |
| :--- | :---: | :---: | :---: | :---: |
| **CRNN (CNN + BiLSTM + CTC)** *(Selected)* | **71.74%**[cite: 2] | 49.78%[cite: 2] | **0.5022**[cite: 2] | **0.1293**[cite: 2] |
| **TrOCR (Vision Transformer)** | 25.80%[cite: 2] | 25.80%[cite: 2] | 0.7410[cite: 2] | 0.5060[cite: 2] |
| **Tesseract OCR (Benchmark)** | 19.50%[cite: 2] | **88.00%**[cite: 2] | 0.8050[cite: 2] | 0.7300[cite: 2] |

* **Key Takeaway:** CRNN demonstrated superior sequence learning for custom Malaysian plate fonts[cite: 2]. Tesseract suffered high CER due to noise sensitivity, while ViT requires larger datasets to avoid underfitting[cite: 2].
* **Integrated Pipeline Score:** Combining **YOLOv8 + CRNN** yielded an overall end-to-end system accuracy of **75%**[cite: 2].

---

## Features

* **Dual Modality Support:** Processes both static images (`.jpg`, `.png`) and real-time video streams (`.mp4`)[cite: 1, 2].
* **Tkinter Graphical User Interface:** User-friendly interface displaying bounding box outputs, detection confidence, cropped plate regions, and recognized text in real-time[cite: 1, 2].
* **Custom Dataset Preprocessing:** Built-in normalization, Albumentations pipeline, and image scaling adjustments optimized for OCR clarity[cite: 2].

---

## Repository Structure

├── assets/                  # Sample output images and system screenshots
├── data/                    # Dataset directory (Images, Annotations, YAML configs)
├── models/
│   ├── detection/           # YOLOv8 and SSD model definitions & checkpoints
│   └── ocr/                 # CRNN, TrOCR (ViT), and Tesseract scripts
├── gui/                     # Tkinter GUI implementation
├── weights/                 # Trained model weights (.pt files)
├── main.py                  # Primary entry point for GUI execution
├── requirements.txt         # Dependencies
└── README.md


---

## Getting Started

### Prerequisites

* Python 3.8 or higher
* CUDA-compatible GPU (recommended for real-time video inference)

### Installation

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/kenn040502/Automatic-Malaysia-Car-License-Plate-Detection-and-Recognition.git](https://github.com/kenn040502/Automatic-Malaysia-Car-License-Plate-Detection-and-Recognition.git)
   cd Automatic-Malaysia-Car-License-Plate-Detection-and-Recognition
Create a Virtual Environment:

Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:

Bash
pip install -r requirements.txt
Run the Application:

Bash
python main.py