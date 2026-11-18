
# 🌱 Modern Agriculture AI – Crop Risk Classification

## 📌 Project Overview

This project uses Machine Learning algorithms to predict crop risk categories based on agricultural and environmental factors.

The main objective is to identify potential crop risks and support farmers in making better agricultural decisions through data-driven predictions.

## 🎯 Problem Statement

Agriculture is affected by several factors, including weather conditions, soil characteristics, irrigation, crop management practices, and technology adoption.

This project develops a classification model that predicts the risk category of agricultural production records, helping identify potential risks and support timely interventions.

## 📂 Dataset Information

- **Dataset:** Modern Agriculture AI Crop Production (2020–2050)
- **Records:** 75,000
- **Features:** 36 columns
- **Target Variable:** `risk_category`
- **Problem Type:** Multiclass Classification

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Jupyter Notebook

## ⚙️ Project Workflow

1. Data Collection and Loading
2. Data Exploration and Understanding
3. Data Cleaning and Preprocessing
4. Handling Missing Values
5. Exploratory Data Analysis (EDA)
6. Outlier Detection
7. Feature Selection
8. Train-Test Split
9. Feature Scaling
10. Categorical Encoding
11. Model Building
12. Cross-Validation
13. Hyperparameter Tuning
14. Model Evaluation and Comparison
15. Prediction using Streamlit Web Application

## 🤖 Machine Learning Algorithms

The following classification algorithms were implemented:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Classifier (SVC)

## 📊 Model Evaluation

Models were evaluated using:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report
- Cross-Validation

Macro F1-score was used during cross-validation and hyperparameter tuning to evaluate performance across all risk categories.

## 🖥️ Web Application

A Streamlit application was developed to provide an interactive interface for crop risk prediction.

To run the application:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure

```text
Modern-Agriculture-AI/
│
├── agriculture classification algm.ipynb
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

## 🚀 Future Improvements

- Improve model performance through additional feature engineering.
- Explore advanced ensemble learning techniques.
- Integrate real-time weather and soil data.
- Develop more advanced agricultural risk prediction features.
- Deploy the application for wider accessibility.

## 👩‍💻 Author

**Geethanjali Munagala**

Machine Learning | Data Science | Artificial Intelligence
