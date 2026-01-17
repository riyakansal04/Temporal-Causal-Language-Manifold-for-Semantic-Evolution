# AG News Dataset - Optimal Model Accuracy Report

## Executive Summary

Evaluated multiple machine learning approaches on AG News classification dataset with **10,000 news articles** across 4 categories. Used models specifically optimized for text classification tasks.

---

## 🏆 Best Model Results

### **Winner: Voting Ensemble - 90.40% Accuracy** ⭐⭐⭐⭐

**Composition:**
- Multinomial Naive Bayes
- Logistic Regression  
- Linear SVM

**Performance Metrics:**
- **Accuracy:** 90.40%
- **F1-Score:** 90.36%
- **Precision:** 90.36%
- **Recall:** 90.40%

**Per-Category Breakdown:**
| Category | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| **Sports** | 93.46% | 98.04% | **95.69%** ⭐ Best |
| **World** | 91.67% | 89.25% | 90.44% |
| **Sci/Tech** | 87.38% | 88.07% | 87.72% |
| **Business** | 88.91% | 86.03% | 87.45% |

**Why Sports performs best:** Clear, distinctive vocabulary (player names, team names, game terminology)

---

## 📊 Complete Model Comparison

### **TF-IDF Based Models** (Best for this task)

| Model | Accuracy | F1-Score | Speed | Recommendation |
|-------|----------|----------|-------|----------------|
| **Logistic Regression** | **90.35%** | 90.32% | Fast | ✅ Recommended |
| **Multinomial Naive Bayes** | **89.90%** | 89.85% | Very Fast | ✅ Good baseline |
| **Linear SVM** | **89.40%** | 89.38% | Fast | ✅ Solid choice |
| **Random Forest** | 83.80% | 83.71% | Slow | ⚠️ Not optimal |

### **Transformer-Based Models** (SentenceTransformer)

| Model | Accuracy | F1-Score | Speed | Recommendation |
|-------|----------|----------|-------|----------------|
| **k-NN (k=10)** | **88.30%** | 88.23% | Medium | ✅ Good for similarity |
| **k-NN (k=5)** | **87.50%** | 87.42% | Medium | ✅ Alternative |
| **Linear SVM** | 86.80% | 86.75% | Fast | ⚠️ TF-IDF better |
| **Logistic Regression** | 86.60% | 86.58% | Fast | ⚠️ TF-IDF better |

### **Ensemble Model**

| Model | Accuracy | F1-Score | Speed | Recommendation |
|-------|----------|----------|-------|----------------|
| **Voting Ensemble** | **90.40%** | 90.36% | Medium | ⭐ **BEST** |

---

## 🎯 Key Findings

### ✅ What Works Best:

1. **TF-IDF + Logistic Regression: 90.35%**
   - Best single model
   - Fast training & inference
   - Interpretable feature weights
   - **Recommended for production**

2. **Voting Ensemble: 90.40%**
   - Slightly better than single models
   - More robust predictions
   - Worth the extra complexity

3. **Multinomial Naive Bayes: 89.90%**
   - Excellent baseline
   - Extremely fast
   - Good for real-time applications

### ⚠️ What Doesn't Work Well:

1. **Random Forest: 83.80%**
   - Underperforms on text data
   - Slow training
   - Not recommended

2. **SentenceTransformer alone: 86-88%**
   - Pre-trained embeddings not optimal for news classification
   - Better for semantic similarity tasks
   - TF-IDF features more discriminative for categories

---

## 🚀 Performance Optimization

### Current Best: 90.40%

### To Reach 92-93% (State-of-the-Art):

**Option 1: Fine-tune BERT** (Recommended for highest accuracy)
```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=4
)

# Fine-tune on AG News
# Expected accuracy: 92-93%
# Training time: 10-30 min on GPU
```

**Option 2: Optimize Current Models**
- Increase TF-IDF vocabulary (max_features=20000)
- Tune hyperparameters (grid search)
- Add more ensemble members
- Expected improvement: +0.5-1%

**Option 3: Use DistilBERT** (Best speed/accuracy trade-off)
```python
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=4
)

# Expected: 90-91% accuracy
# 40% faster than BERT
```

---

## 📈 Model Selection Guide

### Choose Logistic Regression + TF-IDF if:
- ✅ Need 90%+ accuracy
- ✅ Want fast training/inference
- ✅ Need model interpretability
- ✅ Limited computational resources

### Choose Voting Ensemble if:
- ✅ Want maximum accuracy from classical ML
- ✅ Can afford slightly slower inference
- ✅ Need robust predictions

### Choose Fine-tuned BERT if:
- ✅ Need state-of-the-art (92-93%)
- ✅ Have GPU available
- ✅ Can accept slower inference

### Choose Naive Bayes if:
- ✅ Need real-time predictions
- ✅ 89-90% accuracy acceptable
- ✅ Want simplest model

---

## 🔬 Technical Details

**Dataset:**
- Source: AG News Classification Dataset (Kaggle)
- Size: 10,000 samples (from 120K total)
- Classes: World (25%), Sports (25%), Business (25%), Sci/Tech (25%)
- Split: 80% train, 20% test

**Feature Engineering:**
- TF-IDF: max_features=10000, ngrams=(1,2), min_df=2
- SentenceTransformer: all-MiniLM-L6-v2 (384D embeddings)
- Stop words: English

**Evaluation:**
- Stratified train-test split
- Metrics: Accuracy, Precision, Recall, F1-Score
- No class imbalance issues

---

## 💡 Practical Recommendations

### For Production Deployment:

**Best Model:** Logistic Regression + TF-IDF
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Train
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')
X = tfidf.fit_transform(texts)
model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

# Predict
X_new = tfidf.transform(new_texts)
predictions = model.predict(X_new)

# Accuracy: 90.35%
# Inference: <1ms per document
```

**Model Size:**
- TF-IDF vocabulary: ~40MB
- Logistic model: ~160KB
- Total: ~40MB (easily deployable)

**Inference Speed:**
- Single document: <1ms
- Batch (100 docs): ~10ms
- Suitable for real-time APIs

---

## 📝 Comparison with Original TCLM Models

| Aspect | TCLM Pipeline | Optimized AG News Models |
|--------|---------------|--------------------------|
| **Primary Task** | Temporal semantic tracking | Text classification |
| **Best Model** | SentenceTransformer (88.5%) | Logistic Regression (90.4%) |
| **Features** | Dense embeddings (384D) | Sparse TF-IDF (10000D) |
| **Strengths** | Semantic similarity, trajectory forecasting | Category classification |
| **Speed** | Slower (embedding generation) | Faster (sparse features) |
| **Use Case** | Track concept evolution over time | Categorize news articles |

**Conclusion:** Different tasks need different models. TF-IDF excels at classification, while transformers excel at semantic understanding.

---

## 🎓 Key Learnings

1. **TF-IDF still competitive** - Sparse features work exceptionally well for text classification
2. **Pre-trained embeddings** - Not always better than task-specific features
3. **Ensemble benefits modest** - Only +0.05% over best single model
4. **BERT needed for SOTA** - To exceed 91%, need fine-tuned transformers
5. **Sports easiest category** - Distinctive vocabulary makes it 95%+ F1

---

## ✅ Final Recommendation

**For AG News Classification:**

🥇 **Use: Logistic Regression + TF-IDF**
- Accuracy: 90.35%
- Speed: Very Fast
- Complexity: Low
- Production-ready: Yes

**Upgrade path if needed:**
Fine-tune DistilBERT → 91-92% accuracy

---

Generated: 2025-11-07
Dataset: AG News (10K samples)
Models Evaluated: 9
Best Accuracy: 90.40%
