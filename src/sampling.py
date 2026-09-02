from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE


def random_undersample(X_train, y_train):
    """
    Apply Random Undersampling to the training data.

    The minority-to-majority class ratio is set to 0.5.
    """
    sampler = RandomUnderSampler(
        sampling_strategy=0.5,
        random_state=42
    )

    X_resampled, y_resampled = sampler.fit_resample(
        X_train,
        y_train
    )

    return X_resampled, y_resampled


def apply_smote(X_train, y_train):
    """
    Apply SMOTE to the training data.

    The minority-to-majority class ratio is set to 0.5.
    """
    smote = SMOTE(
        sampling_strategy=0.5,
        random_state=42
    )

    X_resampled, y_resampled = smote.fit_resample(
        X_train,
        y_train
    )

    return X_resampled, y_resampled