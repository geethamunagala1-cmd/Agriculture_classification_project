Modern Agriculture AI -- Crop Risk Classification
Project Overview
This project builds an end-to-end machine learning classification
workflow to predict agricultural risk tiers (risk_category) from
farm-record data. The objective is to support targeted, proactive
interventions that may help reduce the risk of crop failure.

The dataset used in the notebook contains 75,000 farm records and 36
columns.
Problem Statement
Build a machine learning classification pipeline to predict agricultural
risk tiers from farm records while considering class imbalance, so that
farms can be grouped by risk and suitable interventions can be
considered.
Target Variable
- Target column: risk_category
- Task: Multiclass classification
The dataset also contains a recommended_action column. It is included
in the source data; check the feature-selection code before using it as
an input because it may encode information closely related to the
target.
Dataset
Expected CSV filename:
modern_agriculture_ai_crop_production_2020_2050.csv
The dataset includes farm, location, crop, soil, irrigation, technology,
production, financial, sustainability and yield-related information.
Examples of columns shown in the notebook include:
- year, region, country, crop
- farm_size_hectares, farm_type, soil_type, irrigation_type
- ai_system_used, technology_stack
- production, cost, revenue and profit fields
- sustainability_score
- yield_forecast_next_year_ton_per_ha
- risk_category
- recommended_action
Place the CSV in the project folder alongside app.py and the notebook,
or update the dataset path in the relevant code.
Workflow
1. Load the CSV using Pandas.
2. Inspect the dataset using head(), tail(), info(),
   describe(), shape and missing-value checks.
3. Replace missing values in ai_system_used with
   None / Unspecified.
4. Check duplicate rows and inspect the distribution of
   risk_category.
5. Visualize risk-category counts and numerical-feature box plots.
6. Separate the features (X) and target (y).
7. Split the data into training and test sets using an 80:20 split,
   random_state=42, and stratification.
8. Calculate IQR bounds on the training data and create a cleaned
   training subset for outlier inspection/handling.
9. Build a preprocessing pipeline:
   - StandardScaler for numerical features
   - OneHotEncoder(handle_unknown="ignore") for categorical
     features
10. Train and evaluate classification models.
11. Evaluate Logistic Regression with stratified 5-fold cross-validation
    using macro F1.
12. Tune Logistic Regression hyperparameters with GridSearchCV.
13. Generate predictions and classification reports for the selected
    model.
Machine Learning Models
The notebook defines these baseline classifiers:
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Classifier (SVC)
- Logistic Regression
The baseline classifiers use class_weight="balanced" to account for
class imbalance. The notebook also evaluates a Random Forest pipeline
and tunes Logistic Regression.
Evaluation
The notebook uses:
- Accuracy
- Precision, recall and F1-score through classification_report
- Confusion matrix (including a normalized confusion matrix for
  Logistic Regression)
- Stratified 5-fold cross-validation with f1_macro
- Grid search with macro F1 scoring
No performance score is stated here because results depend on the exact
data, environment and code execution.
Streamlit Application
The project includes a Streamlit interface in app.py for interacting
with the classification workflow.
Run locally
Open PowerShell or a terminal in the project directory and install the
dependencies:
py -m pip install streamlit pandas numpy scikit-learn
Start the application:
py -m streamlit run app.py
Then open the local address printed in the terminal, usually:
http://localhost:8501
Run a Streamlit application with streamlit run (or
py -m streamlit run), not with Uvicorn. Uvicorn is used to run ASGI
applications such as FastAPI.

Suggested project folder
classification project/
│
├── app.py
├── modern_agriculture_ai_crop_production_2020_2050.csv
├── agriculture classification algm.ipynb
└── README.md
If your CSV is stored elsewhere, update the path expected by app.py.
Requirements
- Python 3
- pandas
- NumPy
- scikit-learn
- Streamlit
- Jupyter Notebook (to run the notebook)
- Matplotlib and Seaborn (for the notebook visualizations)
Install the packages with:
py -m pip install pandas numpy scikit-learn streamlit jupyter matplotlib seaborn
Important Notes
- The notebook's data-loading cell contains a machine-specific Windows
  file path. Change it to the location of the CSV on your computer, or
  use a relative path.
- The notebook creates X_train_clean and y_train_clean after IQR
  filtering, but the later baseline model-fitting cells shown in the
  notebook fit on X_train and y_train. If you intend to train on
  the outlier-filtered data, update those fitting cells consistently
  and evaluate the effect.
- The notebook's feature definition drops risk_category as the
  target but otherwise keeps the remaining columns. Review whether
  columns such as recommended_action should be excluded: if they are
  derived from or strongly tied to the target, including them can
  cause target leakage and produce misleading evaluation results.
- The Streamlit app and notebook should use the same feature columns,
  preprocessing steps and trained model logic. Keep them aligned when
  making changes.
- The dataset is for machine-learning experimentation. Predictions
  should be treated as model estimates, not as a substitute for
  agronomic assessment or field validation.
Future Improvements
- Save and load a fitted preprocessing-and-model pipeline instead of
  retraining on every app launch.
- Add model comparison results and a clearly documented final-model
  selection.
- Validate the model on independent, real-world farm data.
- Review possible target leakage and feature relevance.
- Add input validation and clear explanations for each predicted risk
  class.
