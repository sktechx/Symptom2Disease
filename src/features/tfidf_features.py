
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# CREATE TF-IDF
# ============================================================

def create_tfidf_vectorizer():

    vectorizer = TfidfVectorizer(

        max_features=5000,

        ngram_range=(1, 2),

        min_df=2,

        max_df=0.95,

        sublinear_tf=True
    )

    return vectorizer


# ============================================================
# FIT + TRANSFORM
# ============================================================

def fit_tfidf(
    vectorizer,
    X_train,
    X_test
):

    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )

    X_test_tfidf = vectorizer.transform(
        X_test
    )

    return (
        X_train_tfidf,
        X_test_tfidf
    )