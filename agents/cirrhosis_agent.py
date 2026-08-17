# ==========================================================
# FIX CIRRHOSIS MODEL PACKAGE
# ==========================================================

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


# ==========================================================
# PATHS
# ==========================================================

MODEL_PATH = "/content/LiverAI-MultiAgent/models/cirrhosis/XGBoost_Cirrhosis.pkl"

print("=" * 70)
print("REPAIRING CIRRHOSIS MODEL PACKAGE")
print("=" * 70)


# ==========================================================
# LOAD EXISTING PACKAGE
# ==========================================================

package = joblib.load(MODEL_PATH)

print("\n✓ Package loaded")
print("Package type:", type(package))


# ==========================================================
# EXTRACT MODEL
# ==========================================================

model = package["model"]

print("\nModel type:")
print(type(model))


# ==========================================================
# CORRECT FEATURE NAMES
# ==========================================================

FEATURE_NAMES = [
    "N_Days",
    "Status",
    "Drug",
    "Age",
    "Sex",
    "Ascites",
    "Hepatomegaly",
    "Spiders",
    "Edema",
    "Bilirubin",
    "Cholesterol",
    "Albumin",
    "Copper",
    "Alk_Phos",
    "SGOT",
    "Tryglicerides",
    "Platelets",
    "Prothrombin"
]


# ==========================================================
# CORRECT NUMERICAL FEATURES
# IMPORTANT: Stage IS NOT HERE
# ==========================================================

NUMERICAL_COLUMNS = [
    "N_Days",
    "Age",
    "Bilirubin",
    "Cholesterol",
    "Albumin",
    "Copper",
    "Alk_Phos",
    "SGOT",
    "Tryglicerides",
    "Platelets",
    "Prothrombin"
]


# ==========================================================
# CORRECT CATEGORICAL FEATURES
# ==========================================================

CATEGORICAL_COLUMNS = [
    "Status",
    "Drug",
    "Sex",
    "Ascites",
    "Hepatomegaly",
    "Spiders",
    "Edema"
]


# ==========================================================
# VERIFY MODEL FEATURES
# ==========================================================

model_features = model.get_booster().feature_names

print("\nModel features:")
print(model_features)

print("\nExpected features:")
print(FEATURE_NAMES)


if list(model_features) != FEATURE_NAMES:

    print("\n⚠️ Model feature names/order differs.")

    # If XGBoost has the correct number of features,
    # force the expected names.
    model.get_booster().feature_names = FEATURE_NAMES

    print("✓ Model feature names corrected")

else:

    print("\n✓ Model feature names are correct")


# ==========================================================
# LOAD ORIGINAL DATASET
# ==========================================================

DATASET_PATH = "/content/drive/MyDrive/cirrhosis.csv"

print("\nLoading dataset:")
print(DATASET_PATH)

df = pd.read_csv(DATASET_PATH)

df.columns = df.columns.str.strip()

print("Dataset shape:", df.shape)


# ==========================================================
# PREPARE DATA
# ==========================================================

X = df[FEATURE_NAMES].copy()

y = df["Stage"].copy()


# ==========================================================
# CREATE NEW IMPUTERS
# IMPORTANT:
# THEY ARE FIT ONLY ON INPUT FEATURES
# ==========================================================

numerical_imputer = SimpleImputer(
    strategy="median"
)

categorical_imputer = SimpleImputer(
    strategy="most_frequent"
)


# ==========================================================
# FIT IMPUTERS
# ==========================================================

numerical_imputer.fit(
    X[NUMERICAL_COLUMNS]
)

categorical_imputer.fit(
    X[CATEGORICAL_COLUMNS]
)

print("\n✓ Numerical imputer fitted")
print("Features:")
print(numerical_imputer.feature_names_in_)

print("\n✓ Categorical imputer fitted")
print("Features:")
print(categorical_imputer.feature_names_in_)


# ==========================================================
# REUSE EXISTING ENCODERS
# ==========================================================

encoders = package.get(
    "encoders",
    {}
)

print("\nEncoders:")
print(encoders.keys())


# ==========================================================
# REUSE TARGET ENCODER
# ==========================================================

target_encoder = package.get(
    "target_encoder",
    LabelEncoder()
)

print("\n✓ Target encoder loaded")


# ==========================================================
# CREATE CORRECT PACKAGE
# ==========================================================

new_package = {

    "model": model,

    "feature_names": FEATURE_NAMES,

    "numerical_columns": NUMERICAL_COLUMNS,

    "categorical_columns": CATEGORICAL_COLUMNS,

    "encoders": encoders,

    "target_encoder": target_encoder,

    "numerical_imputer": numerical_imputer,

    "categorical_imputer": categorical_imputer
}


# ==========================================================
# BACKUP OLD PACKAGE
# ==========================================================

BACKUP_PATH = MODEL_PATH.replace(
    ".pkl",
    "_backup.pkl"
)

joblib.dump(
    package,
    BACKUP_PATH
)

print("\n✓ Old package backed up:")
print(BACKUP_PATH)


# ==========================================================
# SAVE CORRECTED PACKAGE
# ==========================================================

joblib.dump(
    new_package,
    MODEL_PATH
)

print("\n" + "=" * 70)
print("✅ CIRRHOSIS PACKAGE REPAIRED")
print("=" * 70)

print("\nSaved:")
print(MODEL_PATH)

print("\nFeature count:")
print(len(FEATURE_NAMES))

print("\nNumerical features:")
print(NUMERICAL_COLUMNS)

print("\nCategorical features:")
print(CATEGORICAL_COLUMNS)
