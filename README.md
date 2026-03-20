# The AI Facility Manager
**A Comparative Benchmarking of Deep Learning vs. Ensemble Methods for Fault Detection in Smart Buildings, Integrated with Automated Technical Support**

[cite_start]**Author:** Lijo Chacko [cite: 5]  
[cite_start]**Program:** MSc Data Science, AI, and Digital Business [cite: 3]  
**University:** GISMA University of Applied Sciences  
[cite_start]**Hardware Platform:** Apple M1 (16GB RAM) [cite: 336-337]

---

## 🏗️ Project Overview
This repository contains the complete, reproducible data science pipeline for my Master's thesis. [cite_start]The project addresses the "Data Rich, Information Poor" (DRIP) problem in modern Facility Management by bridging the gap between numerical anomaly detection and qualitative technical maintenance instructions[cite: 96, 120].

### **Core Research Objectives**
1. [cite_start]**Algorithm Efficiency:** Benchmarking **XGBoost** (Gradient Boosting) against **LSTM** (Long Short-Term Memory) networks for high-frequency HVAC fault detection [cite: 18, 27-28].
2. [cite_start]**Automated Retrieval:** Developing a **RAG (Retrieval-Augmented Generation)** prototype to extract context-aware repair steps from unstructured manufacturer manuals [cite: 19, 30-31].
3. [cite_start]**Digital Business Value:** Evaluating how an integrated AI system reduces **Mean Time to Repair (MTTR)** and **Operational Expenditures (OPEX)** [cite: 20, 134-135].

---

## 📊 Key Results (Chapter 4)
[cite_start]The models were evaluated against the **LBNL Fault Detection Dataset** using a 20% chronological test split (21,024 samples) [cite: 320-322].

| Metric | XGBoost (Baseline) | LSTM (Challenger) | Improvement |
| :--- | :--- | :--- | :--- |
| **Precision (Fault)** | [cite_start]1.00 [cite: 551] | [cite_start]1.00 [cite: 572] | Consistent |
| **Recall (Fault)** | [cite_start]0.80 [cite: 551] | [cite_start]**1.00** [cite: 572] | **+20% Detection** |
| **F1-Score** | [cite_start]0.89 [cite: 551] | [cite_start]**1.00** [cite: 572] | **+11% Accuracy** |
| **Training Time** | [cite_start]0.4245s [cite: 551] | [cite_start]277.2686s [cite: 572] | ~650x Slower |

[cite_start]**RAG Prototype Performance:** The system successfully indexed **5 industrial manuals** into **2,076 discrete semantic chunks** [cite: 601-602]. [cite_start]It demonstrated the ability to map numerical faults to specific repair steps, such as actuator motor replacement instructions[cite: 624, 633].

---

## 📁 Repository Structure
* **`data/`**: 
    * [cite_start]`raw/`: LBNL and ASHRAE CSV telemetry [cite: 24, 224-225].
    * [cite_start]`manuals/`: Corpus of 5 PDF technical manuals (Carrier, Trane, etc.)[cite: 601].
* **`notebooks/`**: 
    * [cite_start]`01_EDA.ipynb`: Analysis of class imbalance (14.29% fault rate) [cite: 391-392].
    * [cite_start]`02_Preprocessing.ipynb`: Z-Score standardization and 3D tensor engineering[cite: 306, 520].
    * [cite_start]`03_XGBoost_Baseline.ipynb`: Ensemble model training and evaluation[cite: 537].
    * [cite_start]`04_LSTM_Model.ipynb`: Deep learning implementation with Early Stopping[cite: 561].
    * [cite_start]`05_RAG_System.ipynb`: FAISS vector database and semantic retrieval testing[cite: 603].
* [cite_start]**`src/`**: Modular Python scripts for time-series windowing and scaling logic[cite: 219].

---

## 🚀 Reproducibility and Setup
This project is optimized for **Apple Silicon (M1/M2/M3)** using `tensorflow-macos`.

