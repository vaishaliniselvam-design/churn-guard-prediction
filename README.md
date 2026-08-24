# Churn Guard: Predicting E-Commerce Customer Attrition Using Machine Learning

An end-to-end machine learning project that predicts customer churn using the Telco Customer Churn dataset. This project covers the complete data science pipeline — from data cleaning to model deployment-ready evaluation.

## 📌 Project Overview

Customer churn (customers leaving a service) is a major cost driver for subscription-based businesses. This project builds classification models to predict which customers are likely to churn, so the business can proactively intervene with retention strategies.

## 📊 Dataset

- **Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers, 21 columns
- **Target variable:** `Churn` (Yes/No)
- **Features:** Demographics (gender, senior citizen, partner, dependents), account info (tenure, contract, payment method, charges), and subscribed services (internet, phone, streaming, tech support, etc.)

## 🛠️ Tools & Libraries

- **Language:** Python (Google Colab)
- **Data handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Imbalance handling:** imbalanced-learn (SMOTE)

## 🔍 Project Workflow

### 1. Data Cleaning
- Converted `TotalCharges` from object to numeric, handled resulting nulls
- Removed duplicate records and the non-predictive `customerID` column

### 2. Exploratory Data Analysis (EDA)
- Analyzed churn distribution (class imbalance ~26.5% churn rate)
- Studied numerical features (tenure, MonthlyCharges, TotalCharges) vs churn using boxplots
- Studied categorical features (Contract, InternetService, PaymentMethod, SeniorCitizen) vs churn using count plots
- Built a correlation heatmap for numerical features

### 3. Feature Engineering
Created 4 new features to improve model signal:
- `TenureGroup` — customer lifecycle stage (0-1yr, 1-2yr, 2-4yr, 4yr+)
- `AvgMonthlySpend` — average spend per month (TotalCharges / tenure)
- `NumServices` — count of add-on services subscribed (engagement score)
- `HighRiskCombo` — flag for month-to-month contract + electronic check payment (high-risk combination)

### 4. Feature Selection
- Applied ANOVA F-test (`SelectKBest`) to rank all features
- Selected the top 20 most predictive features for modeling

### 5. Handling Class Imbalance
- Applied **SMOTE** (Synthetic Minority Oversampling Technique) on the training set to balance churn vs non-churn classes

### 6. Model Building & Comparison
Trained and compared 3 baseline classification models:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

### 7. Hyperparameter Tuning
- Tuned **Random Forest** using `RandomizedSearchCV` (n_estimators, max_depth, min_samples_split, class_weight)
- Tuned **Logistic Regression** using `GridSearchCV` (regularization strength C)

### 8. Model Evaluation
Evaluated final tuned models using:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC score and ROC curve comparison
- Confusion matrix

### 9. Feature Importance & Business Insights
- Extracted top churn-driving features from the tuned Random Forest model
- Translated findings into practical business recommendations

## 📈 Model Results

**Random Forest (Tuned)**
- Accuracy: **0.76** | Precision: **0.543** | Recall: **0.593** | F1 Score: **0.567** | ROC-AUC: **0.813**
- Confusion Matrix: 846 True Negatives, 185 False Positives, 151 False Negatives, 220 True Positives

**Logistic Regression (Tuned)**
- Accuracy: **0.745** | Precision: **0.511** | Recall: **0.787** | F1 Score: **0.62** | ROC-AUC: **0.846**
- Confusion Matrix: 752 True Negatives, 279 False Positives, 79 False Negatives, 292 True Positives

**Best Model:** Logistic Regression achieved the highest ROC-AUC (0.846) and by far the best recall (0.787) — meaning it catches significantly more actual churners than Random Forest, which is critical for a churn-prevention use case where missing a churner is costlier than a false alarm.

## 🔎 Key Findings

Top 5 churn-driving factors (from tuned Random Forest feature importance):

| Rank | Feature | Importance Score |
|------|---------|-------------------|
| 1 | TotalCharges | 0.176 |
| 2 | Tenure | 0.171 |
| 3 | AvgMonthlySpend (engineered) | 0.158 |
| 4 | MonthlyCharges | 0.156 |
| 5 | Contract_Two year | 0.055 |

- **Billing amount and tenure dominate** — TotalCharges, tenure, AvgMonthlySpend, and MonthlyCharges together account for the majority of predictive power, showing that how much and how long a customer has paid is the strongest churn signal
- Customers on a **two-year contract** are far less likely to churn than those on shorter or month-to-month plans
- The engineered feature **HighRiskCombo** (month-to-month + electronic check) and **InternetService_Fiber optic** also rank among the top predictors, confirming these customer segments carry elevated churn risk
- **PaperlessBilling** and **PaymentMethod_Electronic check** further reinforce that billing behavior is closely tied to churn likelihood
- Customers with **TechSupport** and **OnlineSecurity** add-ons show lower churn tendency compared to those without

## 💡 Business Recommendations

1. Since **billing amount and tenure** are the strongest churn signals, monitor customers with rising charges early in their tenure and proactively offer loyalty pricing or check-ins
2. Offer discounted incentives to move month-to-month customers onto **one or two-year contracts**, since contract length strongly reduces churn risk
3. Target the high-risk segment (month-to-month + electronic check payment) with retention offers or a switch-to-autopay incentive
4. Bundle **TechSupport** and **OnlineSecurity** add-ons with Fiber optic plans, since customers with these services churn less
5. Use the model's churn-probability score to flag high-risk customers in real time and route them to a retention team before they cancel

## 📁 Repository Structure

```
churn-guard-prediction/
│
├── Churn_Guard_Project.ipynb   # Full Colab notebook (EDA → Modeling → Evaluation)
└── README.md                   # Project documentation
```

## 🚀 How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. Open the notebook in [Google Colab](https://colab.research.google.com/)
3. Upload the dataset CSV to the Colab session
4. Run all cells in order

## 🙋 About This Project

This project was built as part of the **QSkill Data Science Internship** (Slab 2 – Intermediate), covering the complete machine learning lifecycle: data cleaning, EDA, feature engineering, feature selection, imbalance handling, model comparison, hyperparameter tuning, and evaluation.


.

📦 Dependencies
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
Content
archive (4).zip

