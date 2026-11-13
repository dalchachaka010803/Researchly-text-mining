"""Retrain the deployed model from the dataset.

Rebuilds models/best_model.pickle with the best configuration found in
Question 2 (TF-IDF uni+bigrams -> SMOTE -> Linear SVM), and also reports the
Question 1 baseline (Bag of Words -> Multinomial Naive Bayes) for comparison.

Usage:
    python train_model.py                          # uses ../data/research_abstracts.json
    python train_model.py path/to/research_abstracts.json
"""
import os
import pickle
import sys
import warnings

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocessing import clean

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'data', 'research_abstracts.json')
    print(f"Loading {data_path}")
    df = pd.read_json(data_path).drop_duplicates(subset='Abstract').reset_index(drop=True)
    print(f"{len(df)} unique abstracts")

    print("Cleaning text (takes about a minute)...")
    X = df['Abstract'].apply(clean)
    y = df['Label'].map({'classical': 0, 'ai': 1})

    baseline = Pipeline([
        ('vectorizer', CountVectorizer(min_df=2, ngram_range=(1, 2))),
        ('classifier', MultinomialNB())
    ])
    best = ImbPipeline([
        ('vectorizer', TfidfVectorizer(min_df=2, ngram_range=(1, 2))),
        ('smote', SMOTE(random_state=42)),
        ('classifier', LinearSVC(max_iter=5000))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in [('Baseline: Naive Bayes (Q1)', baseline), ('Best: Linear SVM + SMOTE (Q2)', best)]:
        scores = cross_validate(model, X, y, cv=cv, scoring=['accuracy', 'f1'])
        print(f"{name:32s} 5-fold CV accuracy = {scores['test_accuracy'].mean():.3f}, "
              f"F1 (ai) = {scores['test_f1'].mean():.3f}")

    print("Fitting the best model on all the data...")
    best.fit(X, y)
    os.makedirs(os.path.join(HERE, 'models'), exist_ok=True)
    out = os.path.join(HERE, 'models', 'best_model.pickle')
    with open(out, 'wb') as f:
        pickle.dump(best, f)
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
