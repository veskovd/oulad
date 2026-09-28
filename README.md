# oulad
# Application of Machine Learning Algorithms in Predicting Academic Success

Bachelor thesis project focused on the early prediction of students at risk of academic failure or withdrawal using the **Open University Learning Analytics Dataset (OULAD)**.

The project covers the complete machine learning workflow — from data preparation and feature engineering to model development, evaluation, explainability and application integration.

## Project Overview

The goal of the project was to explore whether student outcomes can be predicted early enough to support timely intervention.

The analysis was based on OULAD, which contains data about **32,593 students across 7 courses and 7 relational tables**.

To simulate an early-warning scenario, student information and activity available within the **first 90 days** were used for feature creation and prediction.

## Machine Learning Workflow

The project included:

- exploratory data analysis and data preprocessing
- aggregation of student activity and assessment data
- feature engineering
- training and comparison of multiple machine learning models
- model evaluation using several classification metrics
- temporal train/test split to better reflect a real-world prediction scenario
- feature selection
- model explainability
- fairness analysis
- additional experiments with deep learning, time-series and multimodal approaches

Six machine learning models were compared, with **Gradient Boosting** achieving the best overall performance.

## Model Explainability

Several explainability approaches were explored:

- **SHAP** for global and local feature importance
- **LIME** for local explanations of individual predictions
- **Counterfactual explanations (DiCE)** to demonstrate which changes in student behavior could affect a prediction

A fairness analysis was also conducted to examine model performance across different student groups.

## Application

A simple application was developed to demonstrate how the prediction model could be used in practice.

The application consists of:

- **FastAPI** backend for model inference
- **Streamlit** user interface
- trained machine learning model
- an **LLM-based component** used to generate personalized explanations and support suggestions based on the model output

The LLM is used for explanation and recommendation generation, while the risk prediction itself is produced by the machine learning model.

## Project Structure

```text
oulad/
│
├── api/
│   └── main.py
│
├── data/
│   └── processed/
│       └── final.csv
│
├── models/
│   ├── final_model.pkl
│   ├── gradient_boosting_model.pkl
│   └── threshold.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_agregacija.ipynb
│   ├── 03_prediktvno_modelovanje.ipynb
│   ├── 04_izbor_atributa.ipynb
│   ├── 05_objasnjenje_odluka.ipynb
│   ├── 06_fairness.ipynb
│   ├── 07_llm.ipynb
│   ├── 08_huggingface.ipynb
│   ├── 09_pytorch.ipynb
│   ├── 10_vremenske_serije.ipynb
│   └── 11_multimodal.ipynb
│
├── streamlit_app.py
└── README.md
