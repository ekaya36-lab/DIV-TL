"""
DIV-TL: Diversity-Guided Composite Pool Sampling with Majority-Only Tomek Links.

This module provides a clean, reusable implementation of the proposed DIV-TL
method for imbalanced binary classification. The method builds a composite
synthetic pool using SMOTE, ADASYN, MWMOTE, and G-SMOTE; selects a diverse
subset of candidates using K-Means-based pool coverage; and then applies
Tomek Links cleaning only to the majority class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_X_y

try:
    import smote_variants as sv
except ImportError as exc:  # pragma: no cover
    sv = None
    _SMOTE_VARIANTS_IMPORT_ERROR = exc
else:
    _SMOTE_VARIANTS_IMPORT_ERROR = None

try:
    from imblearn.under_sampling import TomekLinks
except ImportError as exc:  # pragma: no cover
    TomekLinks = None
    _IMBLEARN_IMPORT_ERROR = exc
else:
    _IMBLEARN_IMPORT_ERROR = None


@dataclass(frozen=True)
class PoolInfo:
    """Metadata about the generated composite synthetic pool."""

    needed_samples: int
    pool_size: int
    selected_size: int
    minority_label: object
    majority_label: object
    n_clusters: int
    removed_majority_by_tomek: int


class DIVTL:
    """
    Diversity-Guided Composite Pool Sampling with Majority-Only Tomek Links.

    Parameters
    ----------
    augmentation_rate : float, default=1.0
        Fraction of the majority-minority gap to fill. ``1.0`` attempts to
        bring the minority class to the majority-class size before cleaning.
    random_state : int, default=42
        Random seed used by the oversamplers, K-Means, and sampling steps.
    max_neighbors : int, default=5
        Upper bound for neighbor-based synthetic sample generators. The actual
        value is safely adapted to the minority-class size.
    max_clusters : int, default=20
        Upper bound for the K-Means clusters used in diversity-guided selection.
    generators : sequence of str, optional
        Composite-pool generators. By default, the method uses
        ``("SMOTE", "ADASYN", "MWMOTE", "G_SMOTE")``.
    tomek_cleaning : bool, default=True
        If True, applies Tomek Links with ``sampling_strategy="majority"`` after
        synthetic samples are selected.

    Notes
    -----
    DIV-TL is intended for binary classification. The minority class is detected
    from the provided ``y`` vector and preserved in the returned labels.
    """

    _GENERATOR_MAP: Dict[str, str] = {
        "SMOTE": "SMOTE",
        "ADASYN": "ADASYN",
        "MWMOTE": "MWMOTE",
        "G_SMOTE": "G_SMOTE",
    }

    def __init__(
        self,
        augmentation_rate: float = 1.0,
        random_state: int = 42,
        max_neighbors: int = 5,
        max_clusters: int = 20,
        generators: Optional[Sequence[str]] = None,
        tomek_cleaning: bool = True,
    ) -> None:
        if augmentation_rate < 0:
            raise ValueError("augmentation_rate must be non-negative.")
        if max_neighbors < 1:
            raise ValueError("max_neighbors must be at least 1.")
        if max_clusters < 1:
            raise ValueError("max_clusters must be at least 1.")

        self.augmentation_rate = float(augmentation_rate)
        self.random_state = int(random_state)
        self.max_neighbors = int(max_neighbors)
        self.max_clusters = int(max_clusters)
        self.generators = tuple(generators or ("SMOTE", "ADASYN", "MWMOTE", "G_SMOTE"))
        self.tomek_cleaning = bool(tomek_cleaning)
        self.pool_info_: Optional[PoolInfo] = None

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a balanced training set using DIV-TL.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Binary class labels.

        Returns
        -------
        X_resampled : ndarray
            Resampled feature matrix.
        y_resampled : ndarray
            Resampled labels using the original label values.
        """
        self._check_dependencies()
        X_checked, y_checked = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
        y_original = np.asarray(y_checked)

        y_internal, minority_label, majority_label = self._encode_binary_labels(y_original)
        n_majority = int(np.sum(y_internal == 0))
        n_minority = int(np.sum(y_internal == 1))
        gap = n_majority - n_minority
        needed = max(0, int(round(self.augmentation_rate * gap)))

        if needed <= 0:
            self.pool_info_ = PoolInfo(
                needed_samples=0,
                pool_size=0,
                selected_size=0,
                minority_label=minority_label,
                majority_label=majority_label,
                n_clusters=0,
                removed_majority_by_tomek=0,
            )
            return X_checked.copy(), y_original.copy()

        pool_X = self._build_composite_pool(X_checked, y_internal, needed)
        if len(pool_X) == 0:
            self.pool_info_ = PoolInfo(
                needed_samples=needed,
                pool_size=0,
                selected_size=0,
                minority_label=minority_label,
                majority_label=majority_label,
                n_clusters=0,
                removed_majority_by_tomek=0,
            )
            return X_checked.copy(), y_original.copy()

        selected_X, n_clusters = self._diversity_select(pool_X, needed)
        selected_y = np.ones(len(selected_X), dtype=int)

        X_aug = np.vstack([X_checked, selected_X])
        y_aug_internal = np.concatenate([y_internal, selected_y])

        before_majority = int(np.sum(y_aug_internal == 0))
        if self.tomek_cleaning:
            X_final, y_final_internal = self._apply_majority_only_tomek(X_aug, y_aug_internal)
        else:
            X_final, y_final_internal = X_aug, y_aug_internal
        after_majority = int(np.sum(y_final_internal == 0))

        y_final = np.where(y_final_internal == 1, minority_label, majority_label)
        self.pool_info_ = PoolInfo(
            needed_samples=needed,
            pool_size=len(pool_X),
            selected_size=len(selected_X),
            minority_label=minority_label,
            majority_label=majority_label,
            n_clusters=n_clusters,
            removed_majority_by_tomek=before_majority - after_majority,
        )
        return X_final, y_final

    def _check_dependencies(self) -> None:
        if sv is None:
            raise ImportError(
                "DIVTL requires smote-variants. Install with: pip install smote-variants"
            ) from _SMOTE_VARIANTS_IMPORT_ERROR
        if TomekLinks is None:
            raise ImportError(
                "DIVTL requires imbalanced-learn. Install with: pip install imbalanced-learn"
            ) from _IMBLEARN_IMPORT_ERROR

    @staticmethod
    def _encode_binary_labels(y: np.ndarray) -> Tuple[np.ndarray, object, object]:
        labels, counts = np.unique(y, return_counts=True)
        if len(labels) != 2:
            raise ValueError(f"DIVTL supports binary classification only; got {len(labels)} classes.")
        minority_label = labels[np.argmin(counts)]
        majority_label = labels[np.argmax(counts)]
        y_internal = np.where(y == minority_label, 1, 0).astype(int)
        return y_internal, minority_label, majority_label

    def _safe_k_neighbors(self, y_internal: np.ndarray) -> Optional[int]:
        minority_count = int(np.sum(y_internal == 1))
        if minority_count < 2:
            return None
        return max(1, min(self.max_neighbors, minority_count - 1))

    def _build_composite_pool(self, X: np.ndarray, y_internal: np.ndarray, needed: int) -> np.ndarray:
        n_majority = int(np.sum(y_internal == 0))
        n_minority = int(np.sum(y_internal == 1))
        gap = n_majority - n_minority
        if gap <= 0:
            return np.empty((0, X.shape[1]), dtype=float)

        # smote-variants uses proportion as the fraction of the class gap to fill.
        proportion = needed / gap
        pool_parts = []
        for generator_name in self.generators:
            synthetic_X = self._generate_with_sv(generator_name, X, y_internal, proportion)
            if len(synthetic_X) > 0:
                pool_parts.append(synthetic_X)

        if not pool_parts:
            return np.empty((0, X.shape[1]), dtype=float)
        return np.vstack(pool_parts)

    def _generate_with_sv(
        self,
        generator_name: str,
        X: np.ndarray,
        y_internal: np.ndarray,
        proportion: float,
    ) -> np.ndarray:
        if generator_name not in self._GENERATOR_MAP:
            raise ValueError(
                f"Unknown generator '{generator_name}'. Supported generators: {sorted(self._GENERATOR_MAP)}"
            )

        safe_k = self._safe_k_neighbors(y_internal)
        if safe_k is None:
            return np.empty((0, X.shape[1]), dtype=float)

        generator_class = getattr(sv, self._GENERATOR_MAP[generator_name])
        candidate_param_sets = [
            {"proportion": proportion, "random_state": self.random_state, "n_neighbors": safe_k, "k_neighbors": safe_k},
            {"proportion": proportion, "random_state": self.random_state, "k_neighbors": safe_k},
            {"proportion": proportion, "random_state": self.random_state, "n_neighbors": safe_k},
            {"proportion": proportion, "random_state": self.random_state},
            {"random_state": self.random_state, "n_neighbors": safe_k, "k_neighbors": safe_k},
            {"random_state": self.random_state, "k_neighbors": safe_k},
            {"random_state": self.random_state, "n_neighbors": safe_k},
            {"random_state": self.random_state},
        ]

        last_error = None
        tried = set()
        for params in candidate_param_sets:
            params = {key: value for key, value in params.items() if value is not None}
            signature = tuple(sorted(params.items()))
            if signature in tried:
                continue
            tried.add(signature)
            try:
                oversampler = generator_class(**params)
                X_res, y_res = oversampler.sample(X, y_internal)
                return self._extract_synthetic_minority(X, X_res, y_res)
            except Exception as exc:  # Different smote-variants classes accept different kwargs.
                last_error = exc
                continue

        # If a generator fails for a dataset, skip it rather than failing the whole method.
        return np.empty((0, X.shape[1]), dtype=float)

    @staticmethod
    def _extract_synthetic_minority(X_original: np.ndarray, X_res: np.ndarray, y_res: np.ndarray) -> np.ndarray:
        original_count = len(X_original)
        new_X = X_res[original_count:]
        new_y = y_res[original_count:]
        return new_X[new_y == 1]

    def _diversity_select(self, pool_X: np.ndarray, needed: int) -> Tuple[np.ndarray, int]:
        rng = np.random.RandomState(self.random_state)
        if needed <= 0 or len(pool_X) == 0:
            return np.empty((0, pool_X.shape[1]), dtype=float), 0

        n_clusters = min(max(2, needed // 5), max(1, len(pool_X) // 2), self.max_clusters)
        n_clusters = max(1, int(n_clusters))

        if n_clusters == 1 or len(pool_X) <= n_clusters:
            replace = len(pool_X) < needed
            selected_idx = rng.choice(len(pool_X), size=needed, replace=replace)
            return pool_X[selected_idx], n_clusters

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(pool_X)

        per_cluster = needed // n_clusters
        remainder = needed % n_clusters
        cluster_sizes = [(int(np.sum(cluster_labels == cluster_id)), cluster_id) for cluster_id in range(n_clusters)]
        cluster_sizes.sort(reverse=True)

        selected_idx = []
        for rank, (_, cluster_id) in enumerate(cluster_sizes):
            cluster_idx = np.where(cluster_labels == cluster_id)[0]
            take = per_cluster + (1 if rank < remainder else 0)
            if take <= 0:
                continue
            take = min(take, len(cluster_idx))
            selected_idx.extend(rng.choice(cluster_idx, size=take, replace=False).tolist())

        while len(selected_idx) < needed:
            selected_idx.append(int(rng.randint(0, len(pool_X))))

        selected_idx = np.asarray(selected_idx[:needed], dtype=int)
        return pool_X[selected_idx], n_clusters

    @staticmethod
    def _apply_majority_only_tomek(X: np.ndarray, y_internal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(np.unique(y_internal)) < 2:
            return X.copy(), y_internal.copy()
        tomek = TomekLinks(sampling_strategy="majority")
        try:
            return tomek.fit_resample(X, y_internal)
        except Exception:
            return X.copy(), y_internal.copy()
