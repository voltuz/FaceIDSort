# 🚀 High-Speed Face Identity Sorter

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)
![GPU Support](https://img.shields.io/badge/GPU-CUDA%20Recommended-orange.svg)

A blazing-fast utility to **sort and group images by facial identity** using cutting-edge deep learning and clustering techniques. Built with `facenet-pytorch`, `DBSCAN`, and a sleek `PyQt6` GUI, this tool makes processing **thousands of face images** efficient, visual, and intuitive.

---

## 📑 Table of Contents
- [✨ Features](#-features)
- [⚙️ Installation](#️-installation)
  - [Using Conda (Recommended)](#using-conda-recommended)
  - [Using Virtualenv + Pip](#using-virtualenv--pip)
- [🚦 Workflow & Usage](#-workflow--usage)
- [⚠️ Limitations & Considerations](#️-limitations--considerations)

---

## ✨ Features

- ⚡ **High-Speed Processing**  
  Utilizes a CUDA-enabled GPU and PyTorch DataLoader for face embedding generation at **500+ images/sec**.

- 🧠 **Two Processing Modes**
  - **Pre-Aligned Mode** *(fastest)* – for tightly cropped face images.
  - **Face Detection Mode** *(slower)* – uses MTCNN for full-scene images.

- 🖼️ **Interactive GUI**  
  Built with `PyQt6`, making it easy to:
  - Select data folders
  - View, inspect, and refine clusters
  - Save final results

- 🔍 **Cluster Editing**  
  Inspect each cluster in detail and manually remove misclassified images before exporting.

- 🧠 **Smart Caching**  
  Embeddings are cached (`face_embeddings_facenet.pkl`) so re-clustering is nearly instant.

- 🗃️ **Project Management**  
  Save your session to JSON and pick up right where you left off.

---

## ⚙️ Installation

### ✅ Requirements
- **Python 3.10+**
- **CUDA-enabled NVIDIA GPU** *(strongly recommended)*
- Dependencies: See `environment.yml` and `requirements.txt`

### 📦 Using Conda (Recommended)

1. **Install Anaconda or Miniconda**  
   [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

2. **Create the Conda Environment**

   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the Environment**

   ```bash
   conda activate face_sorter_env
   ```

4. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   > 🧠 PyTorch should detect your CUDA version. If it doesn't, visit [https://pytorch.org](https://pytorch.org) for the correct command.

---

### 🧪 Using Virtualenv + Pip

1. **Create Virtual Environment**

   ```bash
   python -m venv venv
   ```

2. **Activate the Environment**

   - **Windows**
     ```bash
     .\venv\Scripts\activate
     ```

   - **macOS/Linux**
     ```bash
     source venv/bin/activate
     ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚦 Workflow & Usage

1. **Launch the App**

   ```bash
   python sorter_app.py
   ```

2. **Select Image Folder**  
   Use the GUI to choose the folder containing your images.

3. **Choose Processing Mode**
   - 🖼️ *Pre-Aligned*: For already-cropped face images.
   - 🧑‍🤝‍🧑 *Detect Faces*: For general photos with faces in various positions.

4. **Start Clustering**
   - Generates embeddings using FaceNet
   - Groups similar faces using DBSCAN
   - Shows clusters in the main GUI

5. **Review & Edit Clusters**
   - Use checkboxes to select clusters
   - Use "View / Edit" to inspect images in each cluster
   - Deselect misclassified images manually

6. **Save Results**
   - Choose your preferred output format: `separate folders` or `single folder`
   - Click **Save Selected Clusters** to export

---

## ⚠️ Limitations & Considerations

- **🤖 Model Bias**  
  As with all AI, model accuracy may vary across demographics.

- **🖼️ Image Quality Matters**  
  Blurry or low-res faces will reduce cluster accuracy.

- **🧍 Face Orientation**  
  Best results with frontal or near-frontal faces.

- **🛠️ DBSCAN Tuning**  
  Fine-tune `epsilon` and `min_samples` in "Advanced Settings" for better clustering.  
  - Smaller `epsilon` → tighter, more precise groups  
  - Larger `epsilon` → more lenient grouping

- **📦 "Unclassified" Group (-1)**  
  Outliers that couldn’t be grouped—often poor-quality or unique faces.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

---

## 💬 Contributions & Feedback

Feel free to fork, open issues, or submit PRs. Feedback is always welcome!

---
