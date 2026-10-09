# Diabetes Detector

A machine learning project that predicts whether a patient has diabetes based on diagnostic measurements, using a **Support Vector Machine (SVM)** classifier trained on the PIMA Indians Diabetes Dataset.

> Practice project for learning the basics of data science and machine learning: exploratory data analysis, data preprocessing, model training and evaluation.

## Dataset

**PIMA Indians Diabetes Dataset**: originally from the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), available on [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) (CC0: Public Domain).

- 768 patients (all females, at least 21 years old, of Pima Indian heritage)
- 8 features + 1 target column
- Classes: 500 non-diabetic (0), 268 diabetic (1)

| Column | Description |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration (2-hour oral glucose tolerance test) |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-hour serum insulin (mu U/ml) |
| `BMI` | Body mass index (kg/m²) |
| `DiabetesPedigreeFunction` | Diabetes likelihood based on family history |
| `Age` | Age (years) |
| `Outcome` | Target: 0 = non-diabetic, 1 = diabetic |


## Workflow

1. **Data analysis**: `head()`, `shape`, `describe()`, class distribution with `value_counts()`, feature means per class with `groupby('Outcome').mean()`
2. **Data standardization**: `StandardScaler` to bring all features to the same scale (mean 0, std 1)
3. **Train/test split**: 80% training, 20% test, stratified on `Outcome`
4. **Model training**: `svm.SVC(kernel='linear')`
5. **Evaluation**: accuracy score on training and test data
6. **Predictive system**: predicts the outcome for a new patient from its 8 values

## Results

| Dataset | Accuracy |
|---|---|
| Training | 0.787 |
| Test | 0.773 |

Training and test accuracy are close, so the model is not overfitting.

## Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/runtime-error-r00/Diabetes-Detector.git
cd DiabetesDetector
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

**3. Install the dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the notebook**

Open `notebook/Diabetes_Prediction.ipynb` in VS Code (select the `.venv` kernel) or run:

```bash
jupyter notebook
```

## Disclaimer

This project is for educational purposes only and is **not** a medical diagnostic tool.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.