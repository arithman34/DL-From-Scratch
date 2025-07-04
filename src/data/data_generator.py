from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split


def get_classification_data(num_samples=10000, num_features=10, num_classes=2, n_informative=5, n_redundant=0, n_repeated=0, random_state=None):
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=num_samples,
        n_features=num_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_classes=num_classes,
        random_state=random_state
    )

    return train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)


def get_regression_data(num_samples=10000, num_features=10, n_informative=10, noise=10.0, random_state=None):
    # Generate synthetic regression data
    X, y = make_regression(
        n_samples=num_samples,
        n_features=num_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state
    )

    return train_test_split(X, y, test_size=0.2, random_state=random_state)


def get_clustering_data(num_samples=10000, num_features=2, centers=3, cluster_std=1.0, random_state=None):
    # Generate synthetic clustering data
    X, y = make_blobs(
        n_samples=num_samples,
        n_features=num_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state
    )
    
    return train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
