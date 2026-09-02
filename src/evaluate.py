import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)


def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate classification metrics for fraud detection.

    Parameters:
        y_true: Actual target values.
        y_pred: Predicted class labels.
        y_prob: Predicted probabilities for the positive class (fraud).

    Returns:
        Dictionary containing:
            Precision
            Recall
            F1 Score
            ROC-AUC
            PR-AUC
    """

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob
    )

    # Average Precision is used as the PR-AUC metric
    # to remain consistent with the notebook evaluation.
    pr_auc = average_precision_score(
        y_true,
        y_prob
    )

    return {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
    }


def get_confusion_matrix(y_true, y_pred):
    """
    Return the confusion matrix.

    Parameters:
        y_true: Actual target values.
        y_pred: Predicted class labels.

    Returns:
        Confusion matrix.
    """

    return confusion_matrix(
        y_true,
        y_pred
    )


def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Plot the confusion matrix for a model.

    Parameters:
        y_true: Actual target values.
        y_pred: Predicted class labels.
        model_name: Name of the model.
    """

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Legitimate", "Fraud"],
        cmap="Blues",
    )

    plt.title(
        f"{model_name} - Confusion Matrix"
    )

    plt.tight_layout()
    plt.show()


def plot_precision_recall_curve(y_true, y_prob, model_name):
    """
    Plot the Precision-Recall curve for a model.

    Parameters:
        y_true: Actual target values.
        y_prob: Predicted probabilities for the positive class.

    """

    precision, recall, _ = precision_recall_curve(
        y_true,
        y_prob
    )

    # Use Average Precision to calculate PR-AUC,
    # consistent with the model evaluation.
    pr_auc = average_precision_score(
        y_true,
        y_prob
    )

    plt.figure(figsize=(7, 5))

    plt.plot(
        recall,
        precision,
        label=f"PR-AUC = {pr_auc:.4f}",
        lw=2,
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        f"{model_name} - Precision-Recall Curve"
    )

    plt.legend()
    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained classification model.

    Parameters:
        model: Trained classification model.
        X_test: Test features.
        y_test: Actual test target values.
        model_name: Name of the model.

    Returns:
        metrics: Dictionary containing evaluation metrics.
        y_pred: Predicted class labels.
        y_prob: Predicted fraud probabilities.
    """

    # Generate class predictions
    y_pred = model.predict(
        X_test
    )

    # Generate probabilities for the positive class (fraud)
    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    # Calculate evaluation metrics
    metrics = calculate_metrics(
        y_test,
        y_pred,
        y_prob,
    )

    return metrics, y_pred, y_prob


