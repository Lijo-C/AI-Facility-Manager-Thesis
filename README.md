# The AI Facility Manager
**A Comparative Benchmarking of Deep Learning vs. Ensemble Methods for Fault Detection in Smart Buildings, Integrated with Automated Technical Support**

**Author:** Lijo Chacko  
**Program:** MSc Data Science, AI, and Digital Business  
**University:** GISMA University of Applied Sciences  
**Hardware Platform:** Apple M1

---

## Project Overview
This repository contains the complete, reproducible data science pipeline for my Master's thesis. The project addresses the "Data Rich, Information Poor" (DRIP) problem in modern Facility Management by bridging the gap between numerical anomaly detection and qualitative technical maintenance instructions.

### **Core Research Objectives**
1. **Algorithm Efficiency:** Benchmarking XGBoost (Gradient Boosting) against LSTM (Long Short-Term Memory) networks for high-frequency HVAC fault detection.
2. **Automated Retrieval:** Developing a RAG (Retrieval-Augmented Generation) prototype to extract context-aware repair steps from unstructured manufacturer manuals.
3. **Digital Business Value:** Evaluating how an integrated AI system reduces Mean Time to Repair (MTTR) and Operational Expenditures (OPEX).

---

## Key Results (Chapter 4)
The models were evaluated against the LBNL Fault Detection Dataset using a 20% chronological test split.

| Metric | XGBoost (Baseline) | LSTM (Challenger) | Improvement |
| :--- | :--- | :--- | :--- |
| **Precision (Fault)** | 1.00 | 1.00 | Consistent |
| **Recall (Fault)** | 0.80 | **1.00** | **+20% Detection** |
| **F1-Score** | 0.89 | **1.00** | **+11% Accuracy** |
| **Training Time** | 0.4245s | 277.2686s | ~650x Slower |

**RAG Prototype Performance:** The system successfully indexed 5 industrial manuals into 2,076 discrete semantic chunks. It demonstrated the ability to map numerical faults to specific repair steps, such as actuator motor replacement instructions.

---

## Repository Structure
* **`data/`**: 
    * `raw/`: LBNL and ASHRAE CSV telemetry.
    * `manuals/`: Corpus of 5 PDF technical manuals (Carrier, Trane, etc.).
* **`notebooks/`**: 
    * `01_EDA.ipynb`: Analysis of class imbalance (14.29% fault rate).
    * `02_Preprocessing.ipynb`: Z-Score standardization and 3D tensor engineering.
    * `03_XGBoost_Baseline.ipynb`: Ensemble model training and evaluation.
    * `04_LSTM_Model.ipynb`: Deep learning implementation with Early Stopping.
    * `05_RAG_System.ipynb`: FAISS vector database and semantic retrieval testing.
* **`src/`**: 
    * `data_utils.py`: Modular logic for Z-Score scaling and 3D tensor windowing.
    * `model_utils.py`: Standardized performance reporting functions.
    * `rag_utils.py`: Helper functions for document chunking and FAISS indexing.
    * **Note:** *These scripts are provided for professional reference and modularity. The Jupyter Notebooks in this repository are standalone and contain all necessary logic for direct execution.*

---

## Reproducibility and Setup
This project is optimized for **Apple Silicon (M1/M2/M3)**. 

**Note for Apple Silicon Users:** To utilize the M1 GPU for the 277-second LSTM training as described in Chapter 8, ensure you install both `tensorflow-macos` and `tensorflow-metal` from the `requirements.txt`.