# ML Feature Engineering

A complete, structured repository covering all essential Feature Engineering techniques used in real Machine Learning pipelines — built as part of a hands-on Data Science and ML journey.

---

## 📌 About This Repository

Feature Engineering is the most critical step in any Machine Learning project. A well-engineered feature can improve model accuracy more than switching to a complex algorithm. This repository covers every major technique with real datasets.

---

## 📁 Repository Structure

```
ml-feature-engineering/
│
├── 01_feature_scaling/
│   ├── 01_normalization.py         # MinMaxScaler — scale to 0-1
│   └── 02_standardization.py       # StandardScaler — mean=0, std=1
│
├── 02_feature_encoding/
│   ├── 01_encoding.py              # Label, Ordinal, One-Hot encoding
│   └── 02_column_transformer.py    # Apply transforms to specific columns
│
├── 03_missing_values/
│   ├── 01_numerical_missing.py     # Mean, median, arbitrary imputation
│   └── 02_categorical_missing.py   # Most frequent, constant, indicator
│
├── 04_outlier_removal/
│   ├── 01_iqr_boxplot_method.py    # IQR method with visualization
│   ├── 02_percentile_method.py     # Percentile clipping/winsorization
│   └── 03_zscore_method.py         # Z-score based outlier detection
│
├── 05_feature_transformation/
│   ├── 01_power_transformer.py     # Yeo-Johnson, Box-Cox transforms
│   └── 02_function_transformer.py  # Log, sqrt custom transformations
│
├── 06_feature_construction/
│   ├── 01_mixed_features.py        # Extract numbers from text columns
│   └── 02_feature_splitting.py     # Split one column into many
│
├── 07_datetime_features/
│   └── 01_datetime_extraction.py   # Year, month, day, weekday, is_weekend
│
├── 08_pipelines/
│   └── 01_full_pipeline.py         # Complete ML pipeline end-to-end
│
├── data/
│   ├── titanic.csv                 # Main dataset (survival prediction)
│   ├── weight_height.csv           # Outlier detection practice
│   ├── social_network_ads.csv      # Classification practice
│   ├── data_science_jobs.csv       # Job change prediction
│   ├── concrete_data.csv           # Regression practice
│   ├── covid_toy.csv               # Missing values practice
│   ├── ipl_matches.csv             # Datetime features practice
│   ├── zomato.csv                  # Mixed variables practice
│   └── penguins.csv                # General practice
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🧠 Topics Covered

| # | Topic | Techniques |
|---|-------|-----------|
| 01 | Feature Scaling | MinMaxScaler, StandardScaler |
| 02 | Feature Encoding | LabelEncoder, OrdinalEncoder, OneHotEncoder, ColumnTransformer |
| 03 | Missing Values | Mean, Median, Most Frequent, Constant, Missing Indicator |
| 04 | Outlier Removal | IQR/BoxPlot, Percentile, Z-Score, Winsorization |
| 05 | Transformation | PowerTransformer, FunctionTransformer, Log, Sqrt |
| 06 | Feature Construction | Mixed variables, Feature splitting, Regex extraction |
| 07 | Datetime Features | Year, Month, Day, Weekday, Is_Weekend, Days Since |
| 08 | Pipelines | Full sklearn Pipeline with imputation, encoding, model |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)

---

## 🚀 How to Run

**1. Clone the repository:**
```bash
git clone https://github.com/Arifkhan171/ml-feature-engineering.git
cd ml-feature-engineering
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run any script:**
```bash
python 01_feature_scaling/01_normalization.py
python 08_pipelines/01_full_pipeline.py
```

---

## 👤 Author

**Arif Khan**
Final Year CS Student | University of Loralai | ML • Deep Learning • Agentic AI

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/arif-khan-71a711376)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github)](https://github.com/Arifkhan171)

---

> ⭐ Part of a complete Data Science & AI portfolio — from Feature Engineering to Agentic AI systems.
