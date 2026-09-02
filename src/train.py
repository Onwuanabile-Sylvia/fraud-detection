from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model using balanced class weights.
    """
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier using balanced class weights.
    """
    model = RandomForestClassifier(
        class_weight="balanced",
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model



def train_xgboost(X_train, y_train):
    """
    Train an XGBoost classifier with class-imbalance adjustment.
    """

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = negative_count / positive_count

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    return model