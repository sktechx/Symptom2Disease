from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_vectorizer():
    """
    Create TF-IDF vectorizer.
    """

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )

    return tfidf


def fit_tfidf(
    tfidf,
    X_train,
    X_test
):
    """
    Fit TF-IDF only on training data
    and transform both train and test data.
    """

    X_train_tfidf = tfidf.fit_transform(
        X_train
    )

    X_test_tfidf = tfidf.transform(
        X_test
    )

    return (
        X_train_tfidf,
        X_test_tfidf
    )