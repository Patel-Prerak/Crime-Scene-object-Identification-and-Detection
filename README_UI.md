# 🔍 Crime Scene Evidence Analyzer

**A professional, AI-powered forensic evidence detection and classification system featuring a state-of-the-art UI and advanced computer vision capabilities.**

![Banner](https://img.shields.io/badge/AI-Forensics-blueviolet?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-yellow?style=for-the-badge)

---
## 📖 Overview

The **Crime Scene Evidence Analyzer** is a sophisticated tool designed for forensic professionals. It leverages **YOLOv8** and custom-trained models to detect critical evidence in crime scene imagery. Beyond detection, it offers a premium user experience with real-time statistics, detailed reporting, and a visually immersive interface.

## ✨ Key Features

### 🎨 **Ultra-Professional UI Experience**
*   **Immersive Design:** Animated gradient headers, glassmorphism effects, and consistent theming.
*   **Interactive Elements:** Hover animations on stat boxes, pulsing badges for critical alerts, and smooth transitions.
*   **Responsive Layout:** Optimized for various screen sizes with custom gradient scrollbars.
*   **Visual Badges:**
    *   🔴 **Weapons:** Pulsing red alerts.
    *   🔵 **Persons:** Distinct blue identifiers.
    *   🟣 **Digital Evidence:** Cyan-pink gradients.
    *   🟢 **General Items:** Green-blue gradients.

### 🤖 **Advanced AI Detection**
Powered by a dual-model system for maximum accuracy:
1.  **YOLOv8 Large (Standard):** For robust general object detection.
2.  **Custom Forensic Model:** Specifically trained for:
    *   🩸 Blood Stains
    *   🔫 Weapons (Guns, Knives)
    *   👤 Suspects/Persons
    *   📱 Digital Evidence (Phones, Laptops)
    *   🎒 Personal Items (Backpacks, Handbags)

### 📊 **Comprehensive Analysis**
*   **Real-time Inference:** Instant detection with confidence scores.
*   **Evidence Dashboard:** Summary of total counts, average confidence, and critical findings.
*   **Reporting:**
    *   Export data to **CSV**.
    *   Download **Annotated Images** with bounding boxes.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have **Python 3.8+** installed.

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/paarth-birla/Crime-Scene-Evidence-Classification
    cd Crime-Scene-object-Identification-and-Detection-main
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *(If `requirements.txt` is missing, install manually: `pip install streamlit ultralytics opencv-python pandas pillow numpy`)*

3.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

The application will launch automatically in your default web browser at `http://localhost:8501`.

---

## 📖 Usage Guide

1.  **Upload:** Use the sidebar to upload a crime scene image (JPG, PNG, JPEG).
2.  **Analyze:** Click the **"🔍 Analyze Evidence"** button.
3.  **Explore:**
    *   View the annotated image with clear bounding boxes.
    *   Check the **Evidence Summary** for a statistical breakdown.
    *   Review the granular **Detection Details** table.
4.  **Export:** Use the sidebar or results section to download reports and images.

---

## 🔧 Configuration

*   **Confidence Threshold:** Default is **30%**. Adjustable within the application code to filter low-confidence detections.
*   **Logging:** All detections are logged internally for audit purposes, regardless of the visual threshold.

---

## 🛠️ Technology Stack

*   **Frontend:** [Streamlit](https://streamlit.io/) (Custom CSS & Components)
*   **Deep Learning:** [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
*   **Computer Vision:** OpenCV, PIL
*   **Data Processing:** Pandas, NumPy

---

## ⚠️ Disclaimer

This tool is intended to assist forensic professionals. While it utilizes advanced AI, all automated detections should be verified by qualified human experts.

---

## 📝 License

This project is part of the Crime Scene Evidence Classification system.
