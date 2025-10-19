# Driver Drowsiness Detection using CNN and Computer Vision

An end-to-end **deep learning–based driver monitoring system** that detects **drowsiness in real time** by analyzing **eye closure** and **yawning patterns** using **Convolutional Neural Networks (CNNs)**, **OpenCV**, and **MediaPipe FaceMesh**.  

This project integrates **facial landmark detection**, **feature extraction**, and **multi-modal CNN inference** to alert when a driver shows signs of fatigue.  
Designed for automotive AI safety systems and compatible with embedded implementations for ADAS or smart cabin systems.

---

## 🧩 Project Highlights
- 🧠 Dual-CNN architecture: Separate CNNs for **eye state** (open/closed) and **mouth state** (yawn/no-yawn) detection.  
- 📸 Real-time inference using **MediaPipe FaceMesh** for robust ROI extraction even under low lighting.  
- ⚙️ Fully modular data pipeline (`merge → clean → extract → train → infer`) for easy retraining on new data.  
- 🚀 End-to-end deployment ready: tested with **live webcam feed** and **cabin camera footage**.  
- 🪶 Lightweight inference (<50 ms per frame) — suitable for **in-vehicle edge devices** or **Jetson Nano deployment**.

---


## 🧠 Dataset Sources

This project combines multiple open-source datasets to train and evaluate drowsiness detection models:

| Dataset | Description | Link |
|----------|--------------|------|
| **YawDD (Yawning Detection Dataset)** | Real driver videos annotated for yawning/no-yawn behavior. Used for mouth feature extraction. | [YawDD Dataset (University of Zagreb)](https://www.kaggle.com/datasets/serenaraju/yawdd-dataset) |
| **MRL Eye Dataset** | Contains open and closed eye images collected in controlled lighting. Used for CNN-based eye state classification. | [MRL Eye Dataset on Kaggle](https://www.kaggle.com/datasets/dheerajperumandla/mrl-eye-dataset) |

Only a small **sample subset** is included in `data/sample/` for quick testing.  
To reproduce full training:

```bash
# 1️⃣ Download datasets
mkdir -p data/raw
kaggle datasets download -d serenaraju/yawdd-dataset -p data/raw/
unzip data/raw/yawdd-dataset.zip -d data/raw/yawdd

kaggle datasets download -d dheerajperumandla/mrl-eye-dataset -p data/raw/
unzip data/raw/mrl-eye-dataset.zip -d data/raw/mrleyedataset


