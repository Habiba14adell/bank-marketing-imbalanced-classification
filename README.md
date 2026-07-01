# Bank Marketing Imbalanced Classification

This project analyzes the UCI Bank Marketing dataset and compares several machine learning approaches for predicting whether a client subscribes to a term deposit.

The main modeling challenge is class imbalance: most clients do not subscribe, so accuracy alone is misleading. The cleaned workflow focuses on recall, precision, F1-score, ROC-AUC, and average precision.

## Files

- `bank_marketing_clean.py` - cleaned, reproducible modeling pipeline.
- `Bank-marketing Imbalanced Dataset.ipynb` - original exploratory notebook.
- `phpkIxskf.arff` - Bank Marketing dataset in ARFF format.
- `requirements.txt` - Python dependencies.

## What Was Fixed

The original notebook had a good project direction, but the cleaned version fixes several important ML issues:

- Splits the dataset before resampling to avoid data leakage.
- Uses `stratify=y` so train and test sets preserve the class ratio.
- Applies preprocessing and resampling inside an `imblearn` pipeline.
- Encodes categorical variables instead of dropping them.
- Evaluates with imbalance-aware metrics, not accuracy alone.
- Uses ROC-AUC and average precision from predicted probabilities.
- Compares baseline, class-weighted, SMOTE, and random oversampling strategies.

## Dataset

The data is from the Bank Marketing dataset described by:

Moro, S., Laureano, R., and Cortez, P. (2011). Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology. Proceedings of the European Simulation and Modelling Conference.

The prediction target is `y`, which indicates whether the client subscribed to a term deposit.

## How To Run

Create an environment, install dependencies, and run the cleaned script:

```bash
pip install -r requirements.txt
python bank_marketing_clean.py
```

The script prints model comparison metrics and confusion matrices for each approach.

## Notes

The `duration` feature is included by default because it is part of the assignment dataset and is highly predictive. In a real pre-call marketing system, it should usually be removed because call duration is only known after the call happens.
