# iNeuBytes — Virtual Internship Program (VIIP)
## Artificial Intelligence (Course ID: AIINB20726)

Welcome to the unified repository for the **iNeuBytes Virtual Internship Program (AI)**. This repository contains complete code implementations, experimental results, research reports, and official task specifications (PDFs) for all assigned internship tasks and the major capstone project.

---

## 📑 Repository Structure & Task PDF Specifications

| Task | Title | Directory | Included Specification PDF |
| :--- | :--- | :--- | :--- |
| **Task 1** | Computer Vision using CNN Models | [`Task-1-CNN-ComputerVision/`](./Task-1-CNN-ComputerVision) | [`Task1_CNN_ComputerVision.pdf`](./Task1_CNN_ComputerVision.pdf) |
| **Task 2** | Sentiment Analysis using ML & DL | [`Task-2-Sentiment-Analysis/`](./Task-2-Sentiment-Analysis) | [`Task2_SentimentAnalysis.pdf`](./Task2_SentimentAnalysis.pdf) |
| **Major Project** | Full-Stack AI Recommendation App | [`Major-Project-AI-Recommendation/`](./Major-Project-AI-Recommendation) | [`MajorProject_FullStackAI.pdf`](./MajorProject_FullStackAI.pdf) |

All original problem statement PDF documents are also available in the [`pdfs/`](./pdfs/) folder.

---

## 📌 Project Overview

### 1. Task 1: Computer Vision using CNN Models
- **Objective**: Build baseline AlexNet-style CNN models on CIFAR-10, perform controlled ablation studies (Regularization, Data Augmentation, Optimization, Architecture Depth), and develop a final customized CNN achieving **>70% test accuracy**.
- **Deliverables**: Python code, baseline/final training scripts, master experiment log, evaluation metrics, and research report.

### 2. Task 2: Sentiment Analysis using Machine Learning & Deep Learning
- **Objective**: Build classical NLP baselines (TF-IDF + Logistic Regression & SVM) vs. Deep Learning (LSTM with embeddings & dropout), analyzing performance trade-offs, misclassified edge cases (sarcasm/negation), and F1-score comparisons.
- **Deliverables**: Data pipeline scripts, TF-IDF vs. LSTM evaluation, training curves, confusion matrices, and detailed analysis report.

### 3. Major Project: Full-Stack AI Recommendation Web Application
- **Objective**: Develop and deploy an end-to-end full-stack AI application featuring a TF-IDF & Cosine Similarity recommendation engine with a Flask REST API backend (`/recommend`, `/health`) and a modern interactive UI.
- **Deliverables**: Flask API server, responsive web frontend, Postman test suite collection, setup documentation, and deployment guidelines.

---

## 🚀 Quick Setup & Execution

### Prerequisites
- Python 3.9+
- Node.js (optional for frontend preview)

### Running Task 1 (CNN Computer Vision)
```bash
cd Task-1-CNN-ComputerVision
pip install -r requirements.txt
python main.py
```

### Running Task 2 (Sentiment Analysis)
```bash
cd Task-2-Sentiment-Analysis
pip install -r requirements.txt
python main.py
```

### Running Major Project (AI Recommendation App)
```bash
# Backend
cd Major-Project-AI-Recommendation/backend
pip install -r requirements.txt
python app.py

# Frontend
# Open Major-Project-AI-Recommendation/frontend/index.html in any browser
```

---

## 📄 License & Acknowledgments
Created as part of the **iNeuBytes Virtual Internship Program (VIIP)** in Artificial Intelligence (Course ID: AIINB20726).
