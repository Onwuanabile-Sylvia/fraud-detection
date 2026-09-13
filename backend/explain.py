import shap


# SHAP explainer is created once when this module is loaded.
explainer = None


def get_top_features(
    model,
    transaction,
    feature_names,
    top_n=5
):
    """
    Generate the top SHAP feature contributions
    for a single transaction.
    """

    global explainer

    # Create the SHAP explainer only once
    if explainer is None:
        explainer = shap.TreeExplainer(model)

    # Calculate SHAP values for the transaction
    shap_values = explainer.shap_values(transaction)

    # Get SHAP values for the first transaction
    values = shap_values[0]

    # Store feature contributions
    feature_contributions = {}

    for feature, shap_value in zip(
        feature_names,
        values
    ):
        feature_contributions[feature] = float(shap_value)

    # Sort features by absolute SHAP value
    sorted_features = sorted(
        feature_contributions.items(),
        key=lambda item: abs(item[1]),
        reverse=True
    )

    # Select the top features
    top_features = []

    for feature, contribution in sorted_features[:top_n]:

        if contribution > 0:
            direction = "increased fraud risk"
        elif contribution < 0:
            direction = "reduced fraud risk"
        else:
            direction = "no meaningful effect"

        top_features.append(
            {
                "feature": feature,
                "shap_value": round(contribution, 6),
                "direction": direction
            }
        )

    return top_features