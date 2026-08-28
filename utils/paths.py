# ============================================================
# utils/paths.py
# ============================================================

import os


DRIVE_ROOT = "/content/drive/MyDrive"


# ============================================================
# FATTY LIVER
# ============================================================

FATTY_LIVER_ROOT = os.path.join(
    DRIVE_ROOT,
    "FattyLiver Agent"
)

FATTY_LIVER_DATA = os.path.join(
    FATTY_LIVER_ROOT,
    "DATA"
)


# ============================================================
# FIBROSIS
# ============================================================

FIBROSIS_ROOT = os.path.join(
    DRIVE_ROOT,
    "Fibrosis Agent"
)

FIBROSIS_MODEL = os.path.join(
    FIBROSIS_ROOT,
    "XGBoost_model",
    "xgboost_nafld.pkl"
)


# ============================================================
# CIRRHOSIS
# ============================================================

CIRRHOSIS_ROOT = os.path.join(
    DRIVE_ROOT,
    ".Cirrhosis Agent"
)

CIRRHOSIS_DATA = os.path.join(
    CIRRHOSIS_ROOT,
    "DATA",
    "liver_cirrhosis.csv"
)


# ============================================================
# TUMOR
# ============================================================

TUMOR_ROOT = os.path.join(
    DRIVE_ROOT,
    "Liver CT Image Dataset"
)


# ============================================================
# SEGMENTATION
# ============================================================

SEGMENTATION_ROOT = os.path.join(
    DRIVE_ROOT,
    "archive (2)",
    "image"
)


def check_paths():

    paths = {
        "fatty_liver": FATTY_LIVER_ROOT,
        "fatty_liver_data": FATTY_LIVER_DATA,

        "fibrosis": FIBROSIS_ROOT,
        "fibrosis_model": FIBROSIS_MODEL,

        "cirrhosis": CIRRHOSIS_ROOT,
        "cirrhosis_data": CIRRHOSIS_DATA,

        "tumor": TUMOR_ROOT,

        "segmentation": SEGMENTATION_ROOT,
    }

    print("=" * 70)
    print("LIVERAI PATH CHECK")
    print("=" * 70)

    for name, path in paths.items():

        status = "✓" if os.path.exists(path) else "✗"

        print(f"{status} {name}")
        print(f"  {path}")

    print("=" * 70)
