# 🎓 Recommendation System for Student Performance

A machine learning system that predicts student performance and generates personalized study recommendations based on assessment scores (Quiz, Assignments, Midterm...). The project combines classic ML models (Logistic Regression, Random Forest, SVM) with imbalance-handling techniques (BorderlineSMOTE) and a custom synthetic-sample curation mechanism called **Epiplexity** to filter out low-quality synthetic samples before training.

## 📚 Table of Contents

- [Overview](#-overview)
- [Notebook Structure](#-notebook-structure)
- [Requirements](#-requirements)
- [Dataset Setup & Path Configuration](#-dataset-setup--path-configuration)
- [How to Run](#-how-to-run)
- [Output](#-output)

## 🔎 Overview

The project builds an early-warning and recommendation system across two stages of a course:

- **Stage 1 (20% of course):** Based on Quiz01 + Assignment01 → early warning.
- **Stage 2 (50% of course):** Based on Quiz01 + Assignment01 + Midterm Exam + Assignment02 → mid-course monitoring.

Students are classified under two scenarios:
- **Binary:** G (Good) / W (Weak)
- **Multi-class:** G (Good) / W (Weak) / F (Fail)

After prediction, the system assesses a risk level (Low / Mild / Moderate / High Risk) and generates corresponding study recommendations for each student.

## 📂 Notebook Structure

| Section | Content |
|---|---|
| SECTION 0 | Setup & Imports |
| SECTION 1 | Preprocessing & EDA |
| SECTION 2 | Imbalance handling (BorderlineSMOTE + Epiplexity) |
| SECTION 3 | Model definitions (Logistic Regression, Random Forest, SVM) |
| SECTION 4 | Evaluation helpers |
| SECTION 5 | Binary dataset pipeline (Stage 1, Stage 2, Final comparison) |
| SECTION 6 | Multi-class dataset pipeline (Stage 1, Stage 2, Final comparison) |
| SECTION 7 | Recommendation system (Risk Assessment & Recommendation) |

## ⚙️ Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
```

> The original notebook was developed on **Google Colab** and uses `google.colab.drive` to mount Google Drive. If you run it locally or on another environment (Kaggle, VS Code...), you need to **remove the Drive-mount cell** and update the dataset paths as shown below.

## 📥 Dataset Setup & Path Configuration

The notebook requires two CSV files:

- `Student Performance Prediction-Binary.csv`
- `Student Performance Prediction-Multi.csv`

In the original notebook (Colab), the paths are declared like this:

```python
from google.colab import drive
drive.mount('/content/drive')

# ── Load datasets ───────────────────────────────────────────────────────
binary_path = "/content/drive/MyDrive/Course/thesis/Student Performance Prediction-Binary.csv"
multi_path  = "/content/drive/MyDrive/Course/thesis/Student Performance Prediction-Multi.csv"

df_binary = pd.read_csv(binary_path)
df_multi  = pd.read_csv(multi_path)
```

### 👉 How to update the path to match where you downloaded the dataset

**1. Running on Google Colab:**
Upload the two CSV files to your Google Drive (e.g. into a `MyDrive/dataset/` folder), then update the paths accordingly:

```python
binary_path = "/content/drive/MyDrive/dataset/Student Performance Prediction-Binary.csv"
multi_path  = "/content/drive/MyDrive/dataset/Student Performance Prediction-Multi.csv"
```

**2. Running locally (Jupyter Notebook / VS Code) or on Kaggle:**
Remove the `from google.colab import drive` and `drive.mount(...)` cell, place the two CSV files inside a `data/` folder next to the notebook in the repo, and use relative paths instead:

```python
binary_path = "data/Student Performance Prediction-Binary.csv"
multi_path  = "data/Student Performance Prediction-Multi.csv"

df_binary = pd.read_csv(binary_path)
df_multi  = pd.read_csv(multi_path)
```

Suggested repository structure for GitHub:

```
Recommendation-System-for-Student-Performance/
├── Recommendation_System_for_Student_Performance.ipynb
├── data/
│   ├── Student Performance Prediction-Binary.csv
│   └── Student Performance Prediction-Multi.csv
├── README.md
└── requirements.txt
```

> ⚠️ Note: replace `binary_path` and `multi_path` with the **actual location** where you saved the dataset files on your machine.

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone <your-repo-link>
   cd Recommendation-System-for-Student-Performance
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the two dataset files inside the `data/` folder (or update the paths as described above).
4. Open and run `Recommendation_System_for_Student_Performance.ipynb` cell by cell, from top to bottom.

## 📊 Output

- A comparison table of model performance (Accuracy, Macro-F1, Weak-class Recall) between the original and curated (SMOTE-enriched) datasets.
- The best-performing model automatically selected for each stage (Stage 1 / Stage 2).
- A per-student recommendation report including risk level and a list of suggested actions.
