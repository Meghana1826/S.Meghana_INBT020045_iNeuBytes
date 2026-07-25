# S.Meghana_INBT020045_iNeuBytes
## iNeuBytes Virtual Internship Program (VIIP) — Artificial Intelligence (Course ID: AIINB20726)

**Intern Name:** S. Meghana  
**Intern ID:** INBT020045  
**Course:** Artificial Intelligence (AIINB20726)

---

Welcome to the official internship repository for **S. Meghana (ID: INBT020045)** under the **iNeuBytes Virtual Internship Program**. This repository contains complete source code, datasets, evaluation results, reports, and official task PDF specifications for all assigned tasks and the final major project.

---

## 📑 Repository Structure & Task PDF Specifications

| Task | Title | Code Directory | Official Specification PDF |
| :--- | :--- | :--- | :--- |
| **Task 1** | Computer Vision using CNN Models | [`Task-1-CNN-ComputerVision/`](./Task-1-CNN-ComputerVision) | [`Task1_CNN_ComputerVision.pdf`](./Task1_CNN_ComputerVision.pdf) |
| **Task 2** | Sentiment Analysis using ML & DL | [`Task-2-Sentiment-Analysis/`](./Task-2-Sentiment-Analysis) | [`Task2_SentimentAnalysis.pdf`](./Task2_SentimentAnalysis.pdf) |
| **Major Project** | Full-Stack AI Recommendation Web Application | [`Major-Project-AI-Recommendation/`](./Major-Project-AI-Recommendation) | [`MajorProject_FullStackAI.pdf`](./MajorProject_FullStackAI.pdf) |

📁 All 3 task specification PDFs are also organized inside the [`pdfs/`](./pdfs/) folder.

---

## 📌 Project Overview

### 1. Task 1: Computer Vision using CNN Models
- **Objective**: Implement baseline AlexNet-style CNN on CIFAR-10, conduct ablation studies (Dropout, Batch Normalization, L2 Regularization, Data Augmentation, Optimizers, Architecture Depth), and develop a final customized CNN achieving **>70% test accuracy**.
- **Key Files**: `Task-1-CNN-ComputerVision/main.py`, `train_baseline.py`, `train_final_model.py`, `README.md`.

### 2. Task 2: Sentiment Analysis using Machine Learning & Deep Learning
- **Objective**: Implement classical NLP baselines (TF-IDF + Logistic Regression & SVM) vs. Deep Learning (LSTM with embeddings & dropout), analyzing misclassified edge cases (sarcasm/negation) and performance trade-offs.
- **Key Files**: `Task-2-Sentiment-Analysis/main.py`, `report/SENTIMENT_ANALYSIS_REPORT.md`, `src/`.

### 3. Major Project: Full-Stack AI Recommendation Web Application
- **Objective**: Develop and deploy an end-to-end full-stack AI web application featuring a TF-IDF & Cosine Similarity movie recommendation engine with a Flask REST API backend (`/recommend`, `/health`), modern UI, and Postman API collection.
- **Key Files**: `Major-Project-AI-Recommendation/backend/app.py`, `frontend/index.html`, `postman/api_collection.json`.

---

## 🚀 Quick Start Guide

### Setup Environment
```bash
git clone https://github.com/Meghana1826/S.Meghana_INBT020045_iNeuBytes.git
cd S.Meghana_INBT020045_iNeuBytes
```

### Run Task 1 (CNN Computer Vision)
```bash
cd Task-1-CNN-ComputerVision
pip install -r requirements.txt
python main.py
```

### Run Task 2 (Sentiment Analysis)
```bash
cd Task-2-Sentiment-Analysis
pip install -r requirements.txt
python main.py
```

### Run Major Project (AI Recommendation App)
```bash
# Start Flask Backend Server
cd Major-Project-AI-Recommendation/backend
pip install -r requirements.txt
python app.py

# Open Major-Project-AI-Recommendation/frontend/index.html in a web browser
```

---

## 📄 License & Certification
Submitted by **S. Meghana (INBT020045)** for the **iNeuBytes Virtual Internship Program (VIIP)** in Artificial Intelligence.
