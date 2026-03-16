# The AI Facility Manager
**A Comparative Benchmarking of Deep Learning vs. Ensemble Methods for Fault Detection in Smart Buildings, Integrated with Automated Technical Support**

**Author:** Lijo Chacko  
**Program:** MSc Data Science, AI, and Digital Business  
**University:** GISMA University of Applied Sciences  

## Project Overview
This repository contains the complete, reproducible data science pipeline for my Master's thesis. 
The project addresses the "Data Rich, Information Poor" (DRIP) problem in modern Facility Management by combining quantitative anomaly detection with qualitative Natural Language Processing (NLP).

It investigates three core objectives:

1. **Algorithm Efficiency:** 
Benchmarking XGBoost against Long Short-Term Memory (LSTM) networks for HVAC fault detection.

2. **Automated Retrieval:** 
Developing a Retrieval-Augmented Generation (RAG) prototype to extract actionable troubleshooting steps from unstructured technical manuals.

3. **Integrated Business Value:** 
Evaluating how this system reduces Mean Time to Repair (MTTR) and Operational Expenditures (OPEX).

## Repository Structure
* **`data/`**: Contains data samples and instructions for accessing the full LBNL and ASHRAE datasets.
* **`notebooks/`**: Jupyter notebooks detailing the CRISP-DM workflow, including Exploratory Data Analysis (EDA), Data Preprocessing, and Model Training.
* **`src/`**: Modular Python scripts for data ingestion, evaluation metrics, and the RAG pipeline.
* **`manuals/`**: The corpus of open-source PDF technical manuals used for the RAG vector database.

## Reproducibility and Setup
*`requirements.txt`

## Datasets Used
* **LBNL Fault Detection Dataset:** Real-world, pre-labeled sensor readings from Air Handling Units (AHUs). 
* **ASHRAE Chiller Data:** Operational chiller datasets for performance benchmarking.