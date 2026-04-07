# SMS Spam Detection

A machine learning project that classifies SMS messages as **spam** or **ham (not spam)** using six different classifiers. The project uses TF-IDF vectorization for feature extraction and SMOTE to handle class imbalance.


## Dataset

**SMS Spam Collection Dataset** — 5,572 labelled SMS messages (ham / spam).

Download it from Kaggle and place it at `data/spam.csv`:  
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

| Label | Count |
|-------|-------|
| Ham   | 4,825 |
| Spam  |   747 |

---

## Models Compared

| Model               | Accuracy  |
|---------------------|-----------|
| Naive Bayes         | ~96.5 %   |
| Logistic Regression | ~97.6 %   |
| KNN (k=3)           | ~96.1 %   |
| Decision Tree       | ~96.3 %   |
| **SVM (Linear)**    | **~98.2 %** |
| Random Forest       | ~97.8 %   |

<img width="1048" height="662" alt="Comparison" src="https://github.com/user-attachments/assets/75a3a442-1d98-4ee5-8cf9-3325dd359a7d" />


> Results may vary slightly due to random state.

---

## Setup

### 1 — Clone the repository
```bash
git clone https://github.com/<your-username>/sms-spam-detection.git
cd sms-spam-detection
```

### 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### 3 — Download the dataset
Place `spam.csv` inside the `data/` folder (see Dataset section above).

### 4 — Run the notebook
Open `SMS_Spam_Detection.ipynb` in Jupyter or Google Colab and run all cells.  
Trained artifacts are saved to the `models/` folder.

---

## Quick Predict (CLI)

After training, you can classify a message from the command line:

```bash
python predict.py "Congratulations! You won a free prize, claim now!"
# → SPAM 

python predict.py "Are you free for dinner tonight?"
# → HAM 
```

---

##  Methodology

1. **Load** the SMS Spam Collection dataset.
2. **Encode** labels (`ham → 0`, `spam → 1`).
3. **Split** 80 / 20 into train and test sets.
4. **Vectorize** with TF-IDF (English stop-words removed, `max_df=0.7`).
5. **Balance** the training set with SMOTE (Synthetic Minority Oversampling Technique).
6. **Train & evaluate** six classifiers.
7. **Save** the best-performing model and vectorizer with `joblib`.

---

##  Requirements

See `requirements.txt`. Key packages:

- `scikit-learn`
- `imbalanced-learn`
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `joblib`

---

## License

MIT
