"""Minimal usage example for DIV-TL."""

from divtl import DIVTL
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=8,
    n_redundant=2,
    weights=[0.90, 0.10],
    class_sep=1.0,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42,
)

# Distance-based oversampling methods benefit from scaling.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sampler = DIVTL(augmentation_rate=1.0, random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X_train_scaled, y_train)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_resampled, y_resampled)
y_pred = clf.predict(X_test_scaled)

print("Pool information:", sampler.pool_info_)
print("F1-score:", f1_score(y_test, y_pred))
print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
