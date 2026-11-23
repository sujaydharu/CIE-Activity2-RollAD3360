# CIE 2 – Activity 2: Binary Classification with Logistic Regression

**Student Name:** Sujay Rajendra Dharu  
**Roll Number:** AD3360  
**Course:** Regression and Classification Algorithms (T.Y. B.Tech AI & DS, Semester V)  
**Department:** Artificial Intelligence & Data Science, Zeal College of Engineering and Research, Pune  
**Submission GitHub ID:** [https://github.com/complexiota](https://github.com/complexiota)
---

# ❤️ Heart Disease Risk Prediction using Logistic Regression

## 📌 Project Overview

This project focuses on building a binary classification Machine Learning model using **Logistic Regression** to predict the likelihood of a patient having heart disease based on various clinical parameters.

Developed as part of an RCA (Root Cause Analysis / Machine Learning) project, this implementation emphasizes model interpretability, quantitative evaluation, and ease of deployment using Python and `scikit-learn`.

---

## 📊 Dataset Description

The dataset used is derived from the **UCI Cleveland Heart Disease Dataset**, available on [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).

### Features Breakdown

| No. | Feature | Description |
|---:|:---|:---|
| 1 | `age` | Age in years |
| 2 | `sex` | Sex (1 = Male, 0 = Female) |
| 3 | `cp` | Chest pain type (0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal Pain, 3 = Asymptomatic) |
| 4 | `trestbps` | Resting blood pressure (mm Hg) |
| 5 | `chol` | Serum cholesterol (mg/dl) |
| 6 | `fbs` | Fasting blood sugar greater than 120 mg/dl (1 = True, 0 = False) |
| 7 | `restecg` | Resting electrocardiographic results (0, 1, 2) |
| 8 | `thalach` | Maximum heart rate achieved |
| 9 | `exang` | Exercise-induced angina (1 = Yes, 0 = No) |
| 10 | `oldpeak` | ST depression induced by exercise relative to rest |
| 11 | `slope` | Slope of the peak exercise ST segment |
| 12 | `ca` | Number of major vessels (0–3) colored by fluoroscopy |
| 13 | `thal` | Thalassemia type (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect) |

### Target Variable

| Feature | Description |
|:---|:---|
| `target` | Diagnosis of heart disease (0 = No Disease, 1 = Disease Present) |

---

## ⚙️ Key Technical Workflow

The project follows these major steps:

1. **Data Cleaning & Deduplication:** Removed duplicate rows to reduce the possibility of data leakage between training and testing datasets.
2. **Feature Scaling:** Applied `StandardScaler` to standardize feature ranges (mean = 0, variance = 1), ensuring consistent feature scaling during model optimization.
3. **Model Training:** Trained a `LogisticRegression` baseline classifier using the processed clinical features.
4. **Interpretability Analysis:** Extracted learned coefficients and calculated Odds Ratios using the formula `OR = e^β` to understand the influence of different clinical parameters.
5. **Single-Sample Prediction:** Constructed an end-to-end inference function to evaluate custom patient input values and generate heart disease predictions.

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure Python 3.8 or later is installed on your system.

Install the required Python libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Dataset Setup

1. Clone this repository or download the project source code.
2. Download the heart disease dataset from [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).
3. Place the dataset file named `heart.csv` in the root directory of the project.

### 3. Running the Project

Execute the Python script using the following command:

```bash
python main.py
```

---

## 📈 Model Performance & Results

The Logistic Regression model was evaluated using standard binary classification metrics.

| Evaluation Metric | Score |
|:---|---:|
| **Accuracy** | **80.33%** |
| **Precision** | **80.00%** |
| **Recall** | **84.85%** |
| **F1-Score** | **82.35%** |
| **ROC-AUC Score** | **0.8712** |

### Primary Risk Drivers

The following features were identified as important indicators based on the model's coefficient analysis:

- **Chest Pain Type (`cp`):** Positively associated with the presence of heart disease in the fitted model.
- **Maximum Heart Rate Achieved (`thalach`):** Higher values showed an association with the positive target class.
- **ST Depression (`oldpeak`):** Higher ST depression values acted as a negative indicator in the fitted model.

*Note: These are model-based associations and should not be interpreted as medical diagnoses or proof of causation.*

---

## 📐 Mathematical Foundation

### 1. Sigmoid Activation Function

Logistic Regression maps real-valued linear inputs to probabilities between 0 and 1 using the Sigmoid activation function.

The mathematical representation is:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Where:

$$
z = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n
$$

Here:

- `β₀` = Intercept (bias term)
- `β₁, β₂, ..., βₙ` = Model coefficients
- `X₁, X₂, ..., Xₙ` = Input features
- `σ(z)` = Predicted probability

### 2. Binary Cross-Entropy Loss Function

The model optimizes Binary Cross-Entropy (Log Loss) during training.

$$
\mathcal{L}(\theta) =
-\frac{1}{N}\sum_{i=1}^{N}
\left[
y_i \log(\hat{y}_i)
+
(1-y_i)\log(1-\hat{y}_i)
\right]
$$

Where:

- `N` = Total number of samples
- `yᵢ` = Actual class label
- `ŷᵢ` = Predicted probability
- `L(θ)` = Average Binary Cross-Entropy Loss

### 3. Odds Ratio

Odds Ratios help interpret the influence of individual model coefficients.

$$
OR = e^\beta
$$

An Odds Ratio greater than 1 indicates increased odds of the positive class for a one-unit increase in the corresponding feature, while an Odds Ratio below 1 indicates decreased odds, holding other features constant.

---

## 📜 License

This project is developed for educational purposes as part of the Regression and Classification Algorithms coursework.
