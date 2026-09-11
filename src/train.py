from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, average_precision_score

from catboost import CatBoostClassifier




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

def tune_xgboost(X_train, y_train):
    """
    Tune XGBoost hyperparameters using RandomizedSearchCV.

    Average Precision (PR-AUC) is used as the scoring metric
    because the fraud dataset is highly imbalanced.
    """

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = negative_count / positive_count

    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0]
    }

    pr_auc_scorer = make_scorer(
        average_precision_score,
        response_method="predict_proba"
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=10,
        scoring=pr_auc_scorer,
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_, search.best_score_

def train_catboost(X_train, y_train):
    """
    Train a CatBoost classifier with class-imbalance adjustment.
    """
    model = CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        random_seed=42,
        verbose=0,
        auto_class_weights="Balanced"
    )

    model.fit(X_train, y_train)

    return model