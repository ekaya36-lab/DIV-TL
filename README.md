# DIV-TL

**DIV-TL** (*Diversity Selection with Majority-Only Tomek Links*) is a clean Python implementation of the proposed method described in the manuscript on selection-cleaning-aware composite pool sampling for imbalanced classification.

The method follows three main steps:

1. Build a **composite synthetic pool** using four complementary oversampling generators:
   - SMOTE
   - ADASYN
   - MWMOTE
   - G-SMOTE
2. Apply **diversity-guided selection** to choose synthetic minority candidates from different regions of the pool.
3. Apply **majority-only Tomek Links cleaning** after selection, preserving original and synthetic minority samples while removing overlapping majority-side samples.

## Installation

Clone the repository and install the requirements:

```bash
git clone https://github.com/YOUR_USERNAME/DIV-TL.git
cd DIV-TL
pip install -r requirements.txt
```

Optional editable install:

```bash
pip install -e .
```

## Quick Start

```python
from divtl import DIVTL
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sampler = DIVTL(augmentation_rate=1.0, random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X_train_scaled, y_train)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_resampled, y_resampled)
y_pred = clf.predict(X_test_scaled)

print(balanced_accuracy_score(y_test, y_pred))
print(sampler.pool_info_)
```

## Method Summary

DIV-TL is designed for binary imbalanced classification. It does not rely on a single synthetic sample generator. Instead, it creates a candidate pool from multiple generators and then selects a diverse subset from the pool. Tomek Links cleaning is then applied only to the majority class, so minority samples selected from the pool are preserved.

## Repository Structure

```text
DIV-TL/
├── README.md
├── requirements.txt
├── LICENSE
├── pyproject.toml
├── CITATION.cff
├── divtl/
│   ├── __init__.py
│   └── divtl.py
└── examples/
    └── example_usage.py
```

## Notes

- This implementation expects binary classification labels.
- The minority class is detected automatically from `y`.
- Because the method uses distance-based generators and K-Means, feature scaling is recommended before calling `fit_resample`.
- `smote-variants` may behave slightly differently across versions. The implementation tries multiple compatible parameter configurations for the supported generators.

## Citation

If you use this code in your research, please cite the related manuscript:

```text
Kaya, E., Korkmaz, S., & Sahman, M. A.
Selection and Cleaning-Aware Composite Pool Sampling for Imbalanced Classification.
Manuscript under preparation.
```
