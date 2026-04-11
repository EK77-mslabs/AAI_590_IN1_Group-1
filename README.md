# AI-Driven Insurance Risk Assessment and Policy Recommendation System

This project is part of the AAI-590 Capstone course in the Applied Artificial Intelligence Program at the University of San Diego (USD).

**Project Status:** Completed

---

## Project Objective

The goal of this project is to develop an AI-driven system that predicts the likelihood of insurance claims using structured customer and vehicle data. 

The system assigns a risk score and categorizes users into risk tiers (Low, Medium, High), which are then mapped to appropriate insurance policy recommendations. The project also integrates Explainable AI (XAI) and a Large Language Model (LLM) to provide transparent, human-readable explanations of model decisions.

---

## Contributors

- Eesha Kulkarni  
- Suman Dhankher
- Sudhakar T

---

## Methods Used

- Machine Learning (Logistic Regression, Random Forest, XGBoost)
- Deep Learning (MLP)
- Data Preprocessing & Feature Engineering
- Class Imbalance Handling (Class Weights, SMOTE)
- Model Evaluation & Threshold Tuning
- Explainable AI (SHAP)
- Retrieval-Augmented Generation (RAG)
- LLM-based Explanation System

---

## Technologies

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- SHAP
- Google Gemini API (LLM)
- Jupyter Notebook

---

## Project Description

This project uses the Car Insurance Claim Prediction dataset from Kaggle, consisting of 58,592 records and 43 input features related to policyholders, vehicles, and safety attributes.

The system follows a structured pipeline:

1. Data preprocessing (encoding, scaling, handling imbalance)
2. Model training and comparison (Logistic Regression, Random Forest, XGBoost, MLP)
3. Threshold tuning to optimize recall vs precision trade-offs
4. Risk probability prediction
5. Risk tier assignment (Low, Medium, High)
6. Policy recommendation (Basic, Standard, Premium)
7. Explainability using SHAP
8. LLM-based explanation and Q&A using policy knowledge

The project highlights how machine learning predictions can be combined with interpretable logic and natural language explanations to support real-world decision-making in insurance.

---

## Installation & Usage

1. Clone the repository: 
`git clone https://github.com/EK77-mslabs/AAI_590_IN1_Group-1.git`

2. Navigate to the project folder:
`cd AAI_590_IN1_Group-1`

3. Open the notebook:
`Capstone_Project_Code_Final.ipynb`


4. Run all cells (note: LLM section requires API key)

**Dependencies:** pandas, numpy, scikit-learn, xgboost, tensorflow, shap

## Notes on API Usage

The LLM-based explanation component uses an external API (Google Gemini). The API key has been removed for security reasons.

Outputs shown in the notebook are from a prior successful run. Re-running this section requires a valid API key and may incur usage limits.

---

## License

This project is for academic purposes as part of the AAI-590 course.

---

## Acknowledgments

- University of San Diego (USD)
- Prof. Haisav Chokshi
- Kaggle dataset contributors
