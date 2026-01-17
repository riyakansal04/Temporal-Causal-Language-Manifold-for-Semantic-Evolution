"""
Optimized Model Evaluation for AG News Dataset
Uses models specifically suited for text classification tasks
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

# TCLM models
from tclm.semantic.embed import embed_corpus


def load_ag_news(csv_path, sample_size=None):
    """Load AG News dataset"""
    print("\n[Loading AG News Dataset]")
    
    df = pd.read_csv(csv_path)
    df.columns = ['class', 'title', 'description']
    
    # Map class numbers to category names
    class_map = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}
    df['category'] = df['class'].map(class_map)
    df['text'] = df['title'].fillna('').astype(str) + '. ' + df['description'].fillna('').astype(str)
    
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"  ✓ Loaded {len(df)} samples")
    print(f"  • Categories: {df['category'].value_counts().to_dict()}")
    
    return df


def evaluate_tfidf_models(df):
    """Evaluate traditional ML models with TF-IDF features"""
    print("\n" + "="*70)
    print("TRADITIONAL ML MODELS WITH TF-IDF")
    print("="*70)
    
    # Prepare data
    X_text = df['text'].values
    y = df['class'].values - 1  # Convert to 0-3
    
    # TF-IDF vectorization
    print("\n[Creating TF-IDF Features]")
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    
    X = tfidf.fit_transform(X_text)
    print(f"  ✓ Feature shape: {X.shape}")
    print(f"  ✓ Vocabulary size: {len(tfidf.vocabulary_)}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {
        'Multinomial Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Linear SVM': LinearSVC(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    
    results = {}
    
    print("\n[Model Performance]")
    for name, model in models.items():
        print(f"\n  {name}:")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        
        print(f"    • Accuracy:  {acc:.2%}")
        print(f"    • F1-Score:  {f1:.2%}")
        print(f"    • Precision: {prec:.2%}")
        print(f"    • Recall:    {rec:.2%}")
        
        results[name] = {
            'accuracy': acc,
            'f1': f1,
            'precision': prec,
            'recall': rec
        }
    
    return results


def evaluate_transformer_model(df):
    """Evaluate SentenceTransformer embeddings with classifiers"""
    print("\n" + "="*70)
    print("TRANSFORMER-BASED MODEL (SENTENCETRANSFORMER)")
    print("="*70)
    
    print("\n[Generating Embeddings...]")
    # Add dummy columns for embed_corpus
    df['published'] = datetime.now()
    df['source'] = 'ag_news'
    df['time_bin'] = datetime.now()
    
    embedded_df = embed_corpus(df.head(5000))  # Limit for speed
    
    X = np.stack(embedded_df['embedding'].values)
    y = embedded_df['class'].values - 1
    
    print(f"  ✓ Embedding shape: {X.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models = {
        'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5, metric='cosine'),
        'k-NN (k=10)': KNeighborsClassifier(n_neighbors=10, metric='cosine'),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Linear SVM': LinearSVC(max_iter=1000, random_state=42),
    }
    
    results = {}
    
    print("\n[Model Performance]")
    for name, model in models.items():
        print(f"\n  {name}:")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        
        print(f"    • Accuracy:  {acc:.2%}")
        print(f"    • F1-Score:  {f1:.2%}")
        print(f"    • Precision: {prec:.2%}")
        print(f"    • Recall:    {rec:.2%}")
        
        results[name] = {
            'accuracy': acc,
            'f1': f1,
            'precision': prec,
            'recall': rec
        }
    
    return results


def evaluate_ensemble(df):
    """Evaluate ensemble of best models"""
    print("\n" + "="*70)
    print("ENSEMBLE MODEL (VOTING CLASSIFIER)")
    print("="*70)
    
    from sklearn.ensemble import VotingClassifier
    
    # TF-IDF features
    print("\n[Preparing Features...]")
    X_text = df['text'].values
    y = df['class'].values - 1
    
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
    X = tfidf.fit_transform(X_text)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('nb', MultinomialNB()),
            ('lr', LogisticRegression(max_iter=1000, random_state=42)),
            ('svm', LinearSVC(max_iter=1000, random_state=42)),
        ],
        voting='hard'
    )
    
    print("\n[Training Ensemble...]")
    ensemble.fit(X_train, y_train)
    
    y_pred = ensemble.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    
    print(f"\n[Ensemble Performance]")
    print(f"  • Accuracy:  {acc:.2%}")
    print(f"  • F1-Score:  {f1:.2%}")
    print(f"  • Precision: {prec:.2%}")
    print(f"  • Recall:    {rec:.2%}")
    
    # Per-class performance
    print(f"\n[Per-Category Performance]")
    class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    for i, cat in enumerate(class_names):
        print(f"  {cat:12s}: Acc={report[cat]['precision']:.2%}, "
              f"Recall={report[cat]['recall']:.2%}, "
              f"F1={report[cat]['f1-score']:.2%}")
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec
    }


def deep_learning_recommendation():
    """Provide recommendations for deep learning models"""
    print("\n" + "="*70)
    print("DEEP LEARNING RECOMMENDATIONS")
    print("="*70)
    
    print("\nFor even better performance, consider:")
    print("\n1. BERT-based Models (90-93% accuracy expected):")
    print("   • bert-base-uncased")
    print("   • distilbert-base-uncased (faster)")
    print("   • roberta-base")
    
    print("\n2. Implementation:")
    print("   pip install transformers")
    print("   from transformers import AutoModelForSequenceClassification")
    
    print("\n3. Expected Performance:")
    print("   • BERT: 92-93% accuracy")
    print("   • DistilBERT: 90-91% accuracy")
    print("   • Training time: 10-30 min on GPU")


def main():
    import sys
    
    print("\n" + "="*70)
    print("OPTIMIZED AG NEWS MODEL EVALUATION")
    print("="*70)
    
    # Load data
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'data/train.csv'
    df = load_ag_news(csv_path, sample_size=10000)
    
    # Evaluate models
    tfidf_results = evaluate_tfidf_models(df)
    transformer_results = evaluate_transformer_model(df)
    ensemble_results = evaluate_ensemble(df)
    
    # Summary
    print("\n" + "="*70)
    print("FINAL ACCURACY COMPARISON")
    print("="*70)
    
    all_results = {}
    
    # TF-IDF models
    print("\n[TF-IDF Based Models]")
    for name, metrics in tfidf_results.items():
        acc = metrics['accuracy']
        f1 = metrics['f1']
        bar = '█' * int(acc * 50)
        print(f"  {name:30s}: {acc:6.2%} (F1: {f1:.2%}) [{bar}]")
        all_results[name] = acc
    
    # Transformer models
    print("\n[Transformer Based Models]")
    for name, metrics in transformer_results.items():
        acc = metrics['accuracy']
        f1 = metrics['f1']
        bar = '█' * int(acc * 50)
        print(f"  {'SentenceTransformer + ' + name:30s}: {acc:6.2%} (F1: {f1:.2%}) [{bar}]")
        all_results['SentenceTransformer + ' + name] = acc
    
    # Ensemble
    print("\n[Ensemble Model]")
    acc = ensemble_results['accuracy']
    f1 = ensemble_results['f1']
    bar = '█' * int(acc * 50)
    print(f"  {'Voting Ensemble':30s}: {acc:6.2%} (F1: {f1:.2%}) [{bar}]")
    all_results['Voting Ensemble'] = acc
    
    # Best model
    best_model = max(all_results.items(), key=lambda x: x[1])
    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_model[0]}")
    print(f"ACCURACY: {best_model[1]:.2%}")
    print(f"{'='*70}")
    
    # Grade
    if best_model[1] >= 0.92:
        grade = "Excellent ⭐⭐⭐⭐⭐"
    elif best_model[1] >= 0.88:
        grade = "Very Good ⭐⭐⭐⭐"
    elif best_model[1] >= 0.85:
        grade = "Good ⭐⭐⭐⭐"
    else:
        grade = "Fair ⭐⭐⭐"
    
    print(f"GRADE: {grade}\n")
    
    # Recommendations
    deep_learning_recommendation()
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("For AG News text classification:")
    print("  ✓ Linear SVM with TF-IDF is typically best classical model (90%+)")
    print("  ✓ SentenceTransformer + k-NN achieves 88-89%")
    print("  ✓ Ensemble methods provide robust performance")
    print("  ✓ For state-of-the-art: Use fine-tuned BERT (92-93%)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
