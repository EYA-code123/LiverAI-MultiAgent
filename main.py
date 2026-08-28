```python
# ============================================================
# LiverAI Multi-Agent System
# main.py
# ============================================================

import os
import sys
import pickle
import glob
import traceback
import numpy as np


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = "/content/LiverAI-MultiAgent"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT PATHS
# ============================================================

from utils.paths import PATHS, check_paths


# ============================================================
# IMPORT AGENTS
# ============================================================

from agents.fatty_liver_agent import FattyLiverAgent
from agents.fibrosis_agent import FibrosisAgent
from agents.cirrhosis_agent import CirrhosisAgent
from agents.tumor_classification_agent import TumorClassificationAgent
from agents.liver_segmentation_agent import LiverSegmentationAgent
from agents.clinical_reasoning_agent import ClinicalReasoningAgent


# ============================================================
# ORCHESTRATOR
# ============================================================

from orchestrator.liver_orchestrator import LiverAIOrchestrator


# ============================================================
# MODEL LOADER
# ============================================================

def load_pickle(path):

    if not os.path.exists(path):
        return None

    print(f"Loading model:")
    print(path)

    with open(path, "rb") as f:
        model = pickle.load(f)

    print("✓ Model loaded")
    print(f"  Type: {type(model)}")

    return model


# ============================================================
# SEARCH MODEL FILES
# ============================================================

def find_model_files(root):

    if not os.path.exists(root):
        return []

    patterns = [
        "**/*.pkl",
        "**/*.pickle",
        "**/*.joblib",
        "**/*.h5",
        "**/*.keras",
        "**/*.pth",
        "**/*.pt"
    ]

    files = []

    for pattern in patterns:

        files.extend(
            glob.glob(
                os.path.join(
                    root,
                    pattern
                ),
                recursive=True
            )
        )

    return sorted(
        list(set(files))
    )


# ============================================================
# PRINT MODEL INVENTORY
# ============================================================

def show_model_inventory():

    print()
    print("=" * 80)
    print("LIVERAI MODEL INVENTORY")
    print("=" * 80)

    search_roots = [
        PATHS["fatty_root"],
        PATHS["fibrosis_root"],
        PATHS["cirrhosis_root"],
        PATHS["tumor_root"],
        PATHS["segmentation_root"],
        os.path.join(
            "/content/drive/MyDrive",
            "LiverAI"
        )
    ]

    all_models = []

    for root in search_roots:

        models = find_model_files(root)

        for model in models:

            if model not in all_models:
                all_models.append(model)

    if len(all_models) == 0:

        print("✗ No trained model files found")

    else:

        for model in all_models:
            print(model)

    print()
    print(
        f"TOTAL MODEL FILES: "
        f"{len(all_models)}"
    )

    print("=" * 80)

    return all_models


# ============================================================
# LOAD FATTY LIVER MODEL
# ============================================================

def load_fatty_model():

    print()
    print("=" * 70)
    print("FATTY LIVER MODEL")
    print("=" * 70)

    root = PATHS["fatty_root"]

    candidates = []

    # Search ONLY inside Fatty Liver directory
    candidates.extend(
        find_model_files(root)
    )

    # Keep classical ML files only
    candidates = [
        x for x in candidates
        if x.endswith(
            (".pkl", ".pickle", ".joblib")
        )
    ]

    if not candidates:

        print(
            "⚠ No trained Fatty Liver model found."
        )

        print(
            "Expected a model inside:"
        )

        print(root)

        return None

    # Prefer LightGBM / Random Forest
    preferred = [
        x for x in candidates
        if any(
            key in os.path.basename(x).lower()
            for key in [
                "lightgbm",
                "lgbm",
                "random",
                "forest",
                "fatty"
            ]
        )
    ]

    model_path = (
        preferred[0]
        if preferred
        else candidates[0]
    )

    try:

        return load_pickle(
            model_path
        )

    except Exception as e:

        print(
            f"✗ Fatty model loading failed: {e}"
        )

        return None


# ============================================================
# LOAD FIBROSIS MODEL
# ============================================================

def load_fibrosis_model():

    print()
    print("=" * 70)
    print("FIBROSIS MODEL")
    print("=" * 70)

    path = PATHS["fibrosis_model"]

    if not os.path.exists(path):

        print(
            "✗ Fibrosis model not found:"
        )

        print(path)

        return None

    try:

        return load_pickle(path)

    except Exception as e:

        print(
            f"✗ Fibrosis model loading failed: {e}"
        )

        traceback.print_exc()

        return None


# ============================================================
# LOAD CIRRHOSIS MODEL PACKAGE
# ============================================================

def load_cirrhosis_model():

    print()
    print("=" * 70)
    print("CIRRHOSIS MODEL")
    print("=" * 70)

    root = PATHS["cirrhosis_root"]

    candidates = find_model_files(root)

    candidates = [
        x for x in candidates
        if x.endswith(
            (".pkl", ".pickle", ".joblib")
        )
    ]

    if not candidates:

        print(
            "⚠ No trained Cirrhosis model package found."
        )

        print(
            "The CSV exists, but the trained model "
            "package is not saved yet."
        )

        return None

    preferred = [
        x for x in candidates
        if any(
            key in os.path.basename(x).lower()
            for key in [
                "cirrhosis",
                "xgboost",
                "random"
            ]
        )
    ]

    model_path = (
        preferred[0]
        if preferred
        else candidates[0]
    )

    try:

        package = load_pickle(
            model_path
        )

        if isinstance(package, dict):

            required = [
                "model",
                "feature_names",
                "numerical_columns",
                "categorical_columns",
                "numerical_imputer",
                "categorical_imputer"
            ]

            missing = [
                key
                for key in required
                if key not in package
            ]

            if missing:

                print(
                    "✗ Invalid Cirrhosis model package"
                )

                print(
                    "Missing:",
                    missing
                )

                return None

            return package

        print(
            "✗ Cirrhosis model is not a package/dict"
        )

        return None

    except Exception as e:

        print(
            f"✗ Cirrhosis model loading failed: {e}"
        )

        traceback.print_exc()

        return None


# ============================================================
# LOAD TUMOR MODEL
# ============================================================

def load_tumor_model():

    print()
    print("=" * 70)
    print("TUMOR MODEL")
    print("=" * 70)

    root = PATHS["tumor_root"]

    candidates = find_model_files(root)

    candidates = [
        x for x in candidates
        if x.endswith(
            (".h5", ".keras")
        )
    ]

    if not candidates:

        print(
            "⚠ No trained tumor model found."
        )

        print(
            "Expected .h5 or .keras model."
        )

        return None

    preferred = [
        x for x in candidates
        if any(
            key in os.path.basename(x).lower()
            for key in [
                "efficient",
                "mobilenet",
                "tumor",
                "resnet"
            ]
        )
    ]

    return (
        preferred[0]
        if preferred
        else candidates[0]
    )


# ============================================================
# LOAD SEGMENTATION MODEL
# ============================================================

def load_segmentation_model():

    print()
    print("=" * 70)
    print("SEGMENTATION MODEL")
    print("=" * 70)

    roots = [
        PATHS["segmentation_root"],
        "/content/drive/MyDrive/LiverAI",
        "/content/drive/MyDrive/Liver Segmentation Agent"
    ]

    candidates = []

    for root in roots:

        candidates.extend(
            find_model_files(root)
        )

    candidates = [
        x for x in candidates
        if x.endswith(
            (".pth", ".pt")
        )
    ]

    if not candidates:

        print(
            "⚠ No trained segmentation model found."
        )

        print(
            "Expected a .pth or .pt SegResNet model."
        )

        return None

    preferred = [
        x for x in candidates
        if any(
            key in os.path.basename(x).lower()
            for key in [
                "segresnet",
                "seg",
                "liver"
            ]
        )
    ]

    return (
        preferred[0]
        if preferred
        else candidates[0]
    )


# ============================================================
# INITIALIZE AGENTS
# ============================================================

def create_agents():

    print()
    print("=" * 80)
    print("INITIALIZING LIVERAI AGENTS")
    print("=" * 80)

    agents = {}

    # ========================================================
    # FATTY LIVER
    # ========================================================

    print("\n[1/6] Fatty Liver Agent")

    try:

        fatty_model = load_fatty_model()

        if fatty_model is not None:

            agents["fatty_liver"] = (
                FattyLiverAgent(
                    fatty_model
                )
            )

            print(
                "✓ Fatty Liver Agent initialized"
            )

        else:

            print(
                "⚠ Fatty Liver Agent unavailable"
            )

    except Exception as e:

        print(
            f"✗ Fatty Liver Agent failed: {e}"
        )

    # ========================================================
    # FIBROSIS
    # ========================================================

    print("\n[2/6] Fibrosis Agent")

    try:

        fibrosis_model = (
            load_fibrosis_model()
        )

        if fibrosis_model is not None:

            agents["fibrosis"] = (
                FibrosisAgent(
                    fibrosis_model
                )
            )

            print(
                "✓ Fibrosis Agent initialized"
            )

        else:

            print(
                "✗ Fibrosis Agent unavailable"
            )

    except Exception as e:

        print(
            f"✗ Fibrosis Agent failed: {e}"
        )

    # ========================================================
    # CIRRHOSIS
    # ========================================================

    print("\n[3/6] Cirrhosis Agent")

    try:

        package = (
            load_cirrhosis_model()
        )

        if package is not None:

            agents["cirrhosis"] = (
                CirrhosisAgent(
                    package
                )
            )

            print(
                "✓ Cirrhosis Agent initialized"
            )

        else:

            print(
                "⚠ Cirrhosis Agent unavailable"
            )

    except Exception as e:

        print(
            f"✗ Cirrhosis Agent failed: {e}"
        )

    # ========================================================
    # TUMOR
    # ========================================================

    print("\n[4/6] Tumor Classification Agent")

    try:

        tumor_model = (
            load_tumor_model()
        )

        if tumor_model is not None:

            agents["tumor"] = (
                TumorClassificationAgent(
                    tumor_model
                )
            )

            print(
                "✓ Tumor Classification Agent initialized"
            )

        else:

            print(
                "⚠ Tumor Classification Agent unavailable"
            )

    except Exception as e:

        print(
            f"✗ Tumor Agent failed: {e}"
        )

    # ========================================================
    # SEGMENTATION
    # ========================================================

    print("\n[5/6] Liver Segmentation Agent")

    try:

        segmentation_model = (
            load_segmentation_model()
        )

        if segmentation_model is not None:

            agents["segmentation"] = (
                LiverSegmentationAgent(
                    segmentation_model
                )
            )

            print(
                "✓ Liver Segmentation Agent initialized"
            )

        else:

            print(
                "⚠ Liver Segmentation Agent unavailable"
            )

    except Exception as e:

        print(
            f"✗ Segmentation Agent failed: {e}"
        )

    # ========================================================
    # CLINICAL REASONING
    # ========================================================

    print("\n[6/6] Clinical Reasoning Agent")

    try:

        agents["clinical_reasoning"] = (
            ClinicalReasoningAgent()
        )

        print(
            "✓ Clinical Reasoning Agent initialized"
        )

    except Exception as e:

        print(
            f"✗ Clinical Reasoning Agent failed: {e}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("INITIALIZATION SUMMARY")
    print("=" * 80)

    names = [
        "fatty_liver",
        "fibrosis",
        "cirrhosis",
        "tumor",
        "segmentation",
        "clinical_reasoning"
    ]

    for name in names:

        if name in agents:
            print(f"✓ {name}")
        else:
            print(f"✗ {name}")

    print()
    print(
        f"Initialized: "
        f"{len(agents)}/6"
    )

    print("=" * 80)

    return agents


# ============================================================
# CREATE ORCHESTRATOR
# ============================================================

def create_orchestrator():

    agents = create_agents()

    orchestrator = LiverAIOrchestrator(

        fatty_agent=agents.get(
            "fatty_liver"
        ),

        fibrosis_agent=agents.get(
            "fibrosis"
        ),

        cirrhosis_agent=agents.get(
            "cirrhosis"
        ),

        tumor_agent=agents.get(
            "tumor"
        ),

        segmentation_agent=agents.get(
            "segmentation"
        ),

        clinical_reasoning_agent=agents.get(
            "clinical_reasoning"
        )
    )

    return orchestrator


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\nLiverAI Multi-Agent System"
    )

    check_paths()

    show_model_inventory()

    orchestrator = (
        create_orchestrator()
    )

    print()
    print(
        "Orchestrator status:"
    )

    print(
        orchestrator.get_status()
    )
```
