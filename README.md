# 🎓 Intelligent Student Feedback Analysis CRS
### Classification, Clustering, NER, RAG Chatbot & Recommender System

---

## 👥 Team

**Group 22**

| Name                           | Student ID |
|--------------------------------|------------|
| Udi Bhasin                     | 300475136  |
| Madhumitha Sri Murali          | 300456023  |
| Shruthi Perumalsamy Gurunathan | 300408332  |

## 🌐 Live Demo

- **Streamlit App:** https://group22finalproject-ifzeusf6wtqciggj3fqwvh.streamlit.app/
- **Google Drive:** https://drive.google.com/drive/folders/1j45XNrkgoHf4_YCCmTczuNdYxivsMbnE?usp=sharing

---

## 📁 Project Structure

```
Group22_FinalProject/
├── streamlit_app.py          ← Streamlit demo interface
├── requirements.txt          ← Streamlit deployment dependencies
├── setup.sh                  ← SpaCy model setup for deployment
├── notebook/
│   └── your_notebook.ipynb   ← Main notebook (run in Google Colab)
├── data/
│   ├── rmp_final_dataset.csv ← Main dataset (also available in the Drive Link)
│   ├── rmp_final_with_features.csv
│   └── error_analysis_full.csv
├── models/
│   ├── best_model_lr.joblib
│   └── tfidf_vectorizer.joblib
└── outputs/
    ├── project_summary.json
    └── images/               ← All visualizations
```

---

## 🚀 How to Run

### Step 1 — Mount Google Drive in Colab
All dataset and output files are available in the Google Drive folder linked above.
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2 — Install Dependencies
```bash
!pip install spacy scikit-learn imbalanced-learn scikit-surprise streamlit
!pip install sentence-transformers wordcloud plotly
!python -m spacy download en_core_web_sm
!pip install -q "numpy==1.26.4" scikit-surprise matplotlib --force-reinstall
```

### Step 3 — Run Main Notebook
Open the notebook from `notebook/` in Google Colab and run each section in order.

### Step 4 — Run Streamlit Locally (optional)
```bash
streamlit run streamlit_app.py
```

---

## 🌍 Deployment

The Streamlit app is deployed on **Streamlit Community Cloud:**
- Connected directly to this GitHub repository
- Auto-redeploys on every git push to `main`
- Python version locked to **3.11** for scikit-surprise compatibility
- spaCy `en_core_web_sm` model installed via `setup.sh` on startup

---

## 📊 Project Sections

| Section | What it does | Rubric Coverage |
|---------|-------------|-----------------|
| 0 | Install & Import | Setup |
| 1 | Load Dataset | Data Sources |
| 2 | Deep Data Cleaning | Data Preparation 2.5% |
| 3 | NER + Aspect Extraction (SpaCy) + Feature Engineering | Text Feature Engineering 3% |
| 4 | EDA Visualizations | Visualization 3% |
| 5 | TF-IDF + SMOTE | Data Preparation + Feature Engineering |
| 6 | 5 Classification Models | Classification 3% + Evaluation 4% |
| 7 | Error Analysis | Error Analysis 3% |
| 8 | KMeans + DBSCAN + Agglomerative + t-SNE | Clustering 3% + Visualization 3% |
| 9 | Recommender System (SVD, KNN, NMF) | Recommender 3% |
| 10 | RAG Chatbot | Innovativeness 1% |
| 11 | Save outputs | Report Quality |
| Streamlit | Live demo interface | Visualization 3% |

---

## 🗂️ Dataset

- **Source:** RateMyProfessor (Mendeley Data, Dr. Jibo He)
- **Link:** https://data.mendeley.com/datasets/fvtfjyvw7d/2
- **Size:** 4,164 reviews across 5 departments
- **Departments:** English, Mathematics, Psychology, Biology, History
- **Sentiment labels:** Positive (≥4★), Neutral (3–3.5★), Negative (<3★)
- **Class distribution:** Positive 2,489 | Negative 1,153 | Neutral 522
- **Imbalance handled with:** SMOTE during model training

---

## 🤖 Models Used

### Classification (5 models)

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline, best overall |
| Linear SVM | Strong on sparse TF-IDF |
| Naive Bayes | Fast, probabilistic |
| Random Forest | Ensemble, handles non-linearity |
| Gradient Boosting | Best for complex patterns |

### Clustering (3 algorithms)

| Algorithm | Notes |
|-----------|-------|
| KMeans (k=5) | Main clustering, justified by Elbow + Silhouette |
| DBSCAN | Detects noise/outliers |
| Agglomerative | Hierarchical, compared via Silhouette |

### Recommender (3 algorithms)

| Algorithm | Notes |
|-----------|-------|
| SVD | Matrix factorization, best RMSE |
| KNN User-Based | Collaborative filtering |
| NMF | Non-negative matrix factorization |

---

## 💬 RAG Chatbot

The chatbot uses **Retrieval-Augmented Generation (RAG):**
1. User enters a natural language query
2. Query is vectorized using TF-IDF
3. Cosine similarity retrieves the most relevant reviews
4. Structured response is generated with sentiment breakdown + examples
5. Professor recommendations added if requested

**Example queries:**
- *"What do students complain about exams?"*
- *"Recommend a good mathematics professor"*
- *"What are the most common issues with grading in English?"*

---

## 📈 Classification Results

| Model | Accuracy | F1 Macro | F1 Weighted |
|-------|----------|----------|-------------|
| Logistic Regression | 0.77 | 0.68 | 0.78 |
| Linear SVM | 0.77 | 0.66 | 0.77 |
| Naive Bayes | 0.73 | 0.62 | 0.74 |
| Random Forest | 0.77 | 0.57 | 0.73 |
| Gradient Boosting | 0.76 | 0.60 | 0.74 |

### Per-Class Results — Logistic Regression (Best Model)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 0.76 | 0.81 | 0.78 | 231 |
| Neutral | 0.36 | 0.46 | 0.41 | 104 |
| Positive | 0.90 | 0.81 | 0.85 | 497 |
| Macro Avg | 0.67 | 0.70 | 0.68 | 832 |
| Weighted Avg | 0.79 | 0.77 | 0.78 | 832 |

- **Clustering:** KMeans Silhouette ~0.08–0.12 (typical for text data)
- **Recommender:** SVD RMSE ~0.85–1.1

---

## 🔍 NER & Aspect Extraction

Using SpaCy `en_core_web_sm`:
- **NER:** Extracts PERSON, ORG, GPE entities from reviews
- **Aspect extraction:** Identifies 6 aspects: `exams`, `grading`, `workload`, `lectures`, `professor`, `attendance`
- **Validation:** Compared against binary tag columns (ground truth)

---

## 📦 Requirements

```
pandas numpy matplotlib seaborn plotly wordcloud
scikit-learn imbalanced-learn
spacy (+ en_core_web_sm model)
scikit-surprise (notebook only — not needed for Streamlit deployment)
streamlit joblib scipy
```
