# ============================================================
# LiverAI - paths.py
# ============================================================

import os


# ============================================================
# GOOGLE DRIVE
# ============================================================

DRIVE_ROOT = "/content/drive/MyDrive"


# ============================================================
# DATASETS / MODELS
# ============================================================

PATHS = {

    # --------------------------------------------------------
    # FATTY LIVER
    # --------------------------------------------------------
    "fatty_root":
        os.path.join(
            DRIVE_ROOT,
            "FattyLiver Agent"
        ),

    "fatty_data":
        os.path.join(
            DRIVE_ROOT,
            "FattyLiver Agent",
            "DATA"
        ),

    # --------------------------------------------------------
    # FIBROSIS
    # --------------------------------------------------------
    "fibrosis_root":
        os.path.join(
            DRIVE_ROOT,
            "Fibrosis Agent"
        ),

    "fibrosis_model":
        os.path.join(
            DRIVE_ROOT,
            "Fibrosis Agent",
            "XGBoost_model",
            "xgboost_nafld.pkl"
        ),

    # --------------------------------------------------------
    # CIRRHOSIS
    # --------------------------------------------------------
    "cirrhosis_root":
        os.path.join(
            DRIVE_ROOT,
            ".Cirrhosis Agent"
        ),

    "cirrhosis_data":
        os.path.join(
            DRIVE_ROOT,
            ".Cirrhosis Agent",
            "DATA",
            "liver_cirrhosis.csv"
        ),

    # --------------------------------------------------------
    # TUMOR
    # --------------------------------------------------------
    "tumor_root":
        os.path.join(
            DRIVE_ROOT,
            "Liver CT Image Dataset"
        ),

    # --------------------------------------------------------
    # SEGMENTATION
    # --------------------------------------------------------
    "segmentation_root":
        os.path.join(
            DRIVE_ROOT,
            "archive (2)",
            "image"
        ),

    # --------------------------------------------------------
    # OPTIONAL MODEL DIRECTORIES
    # --------------------------------------------------------

    "models_root":
        os.path.join(
            DRIVE_ROOT,
            "LiverAI"
        ),
}


# ============================================================
# CHECK PATH
# ============================================================

def check_path(path):
    return os.path.exists(path)


# ============================================================
# CHECK ALL PATHS
# ============================================================

def check_paths(verbose=True):

    results = {}

    for name, path in PATHS.items():

        exists = os.path.exists(path)

        results[name] = path if exists else None

        if verbose:

            if exists:
                print(f"✓ {name}")
                print(f"    {path}")

            else:
                print(f"✗ {name}")
                print(f"    {path}")

    return results


# ============================================================
# GET PATH
# ============================================================

def get_path(name):

    if name not in PATHS:
        raise KeyError(
            f"Unknown path: {name}"
        )

    return PATHS[name]
