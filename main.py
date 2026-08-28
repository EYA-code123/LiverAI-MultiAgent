# ============================================================
# LIVERAI - REPAIR main.py
# ============================================================

import os
import sys
import pickle
import joblib
import traceback

# ============================================================
# PROJECT
# ============================================================

PROJECT_DIR = "/content/LiverAI-MultiAgent"
DRIVE_BASE = "/content/drive/MyDrive"

LIVERAI_DIR = os.path.join(
    DRIVE_BASE,
    "LiverAI"
)

# ============================================================
# REAL GOOGLE DRIVE PATHS
# ============================================================

FATTY_ROOT = os.path.join(
    DRIVE_BASE,
    "FattyLiver Agent"
)

FATTY_DATA = os.path.join(
    FATTY_ROOT,
    "DATA"
)

FIBROSIS_ROOT = os.path.join(
    DRIVE_BASE,
    "Fibrosis Agent"
)

FIBROSIS_MODEL = os.path.join(
    FIBROSIS_ROOT,
    "XGBoost_model",
    "xgboost_nafld.pkl"
)

CIRRHOSIS_ROOT = os.path.join(
    DRIVE_BASE,
    ".Cirrhosis Agent"
)

CIRRHOSIS_DATA = os.path.join(
    CIRRHOSIS_ROOT,
    "DATA",
    "liver_cirrhosis.csv"
)

TUMOR_ROOT = os.path.join(
    DRIVE_BASE,
    "Liver CT Image Dataset"
)

SEGMENTATION_ROOT = os.path.join(
    DRIVE_BASE,
    "archive (2)",
    "image"
)

# ============================================================
# PROJECT PATH
# ============================================================

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

os.chdir(PROJECT_DIR)

print("=" * 80)
print("LIVERAI MULTI-AGENT SYSTEM")
print("=" * 80)
print("PROJECT:")
print(PROJECT_DIR)
print("GOOGLE DRIVE:")
print(DRIVE_BASE)
print("=" * 80)


# ============================================================
# CHECK PATHS
# ============================================================

def check_paths():

    paths = {
        "fatty_root": FATTY_ROOT,
        "fatty_data": FATTY_DATA,
        "fibrosis_root": FIBROSIS_ROOT,
        "fibrosis_model": FIBROSIS_MODEL,
        "cirrhosis_root": CIRRHOSIS_ROOT,
        "cirrhosis_data": CIRRHOSIS_DATA,
        "tumor_root": TUMOR_ROOT,
        "segmentation_root": SEGMENTATION_ROOT
    }

    print()
    print("=" * 70)
    print("CHECKING PATHS")
    print("=" * 70)

    for name, path in paths.items():

        if os.path.exists(path):

            print(f"✓ {name}")
            print(f"    {path}")

        else:

            print(f"✗ {name}")
            print(f"    NOT FOUND: {path}")

    print("=" * 70)

    return paths


# ============================================================
# FIND MODEL FILES
# ============================================================

def find_model_files():

    print()
    print("=" * 80)
    print("SEARCHING FOR TRAINED MODEL FILES")
    print("=" * 80)

    model_extensions = (
        ".pkl",
        ".pickle",
        ".joblib",
        ".keras",
        ".h5",
        ".pt",
        ".pth",
        ".onnx"
    )

    found_models = []

    search_roots = [
        FATTY_ROOT,
        FIBROSIS_ROOT,
        CIRRHOSIS_ROOT,
        TUMOR_ROOT,
        os.path.join(DRIVE_BASE, "archive (2)")
    ]

    for root in search_roots:

        if not os.path.exists(root):
            continue

        for current_root, dirs, files in os.walk(root):

            for filename in files:

                if filename.lower().endswith(model_extensions):

                    full_path = os.path.join(
                        current_root,
                        filename
                    )

                    found_models.append(full_path)

                    print(full_path)

    print()
    print("=" * 80)
    print(f"TOTAL MODEL FILES FOUND: {len(found_models)}")
    print("=" * 80)

    return found_models


# ============================================================
# MODEL INVENTORY
# ============================================================

def show_model_inventory():

    print()
    print("=" * 80)
    print("LIVERAI MODEL INVENTORY")
    print("=" * 80)

    models = find_model_files()

    if len(models) == 0:

        print("NO TRAINED MODEL FILES FOUND")

    else:

        for i, model in enumerate(models, 1):

            print(
                f"{i}. {model}"
            )

    print("=" * 80)

    return models


# ============================================================
# SHOW MODEL PATHS
# ============================================================

def show_model_paths():

    print()
    print("=" * 80)
    print("MODEL PATHS")
    print("=" * 80)

    paths = {
        "Fatty Liver": None,
        "Fibrosis": FIBROSIS_MODEL,
        "Cirrhosis": None,
        "Tumor": None,
        "Segmentation": None
    }

    for name, path in paths.items():

        if path is not None and os.path.exists(path):

            print(
                f"✓ {name:<20}: {path}"
            )

        else:

            print(
                f"✗ {name:<20}: MODEL NOT FOUND"
            )

    print("=" * 80)

    return paths


# ============================================================
# LOAD PICKLE
# ============================================================

def load_pickle_model(path):

    if path is None:
        raise FileNotFoundError("Model path is None")

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Model not found:\n{path}"
        )

    print()
    print("Loading model:")
    print(path)

    try:

        model = joblib.load(path)

    except Exception:

        with open(path, "rb") as f:

            model = pickle.load(f)

    print("✓ Model loaded")
    print("Type:", type(model))

    return model


# ============================================================
# IMPORT AGENTS
# ============================================================

from agents.fatty_liver_agent import FattyLiverAgent

from agents.fibrosis_agent import FibrosisAgent

from agents.cirrhosis_agent import CirrhosisAgent

from agents.tumor_classification_agent import (
    TumorClassificationAgent
)

from agents.liver_segmentation_agent import (
    LiverSegmentationAgent
)

from agents.clinical_reasoning_agent import (
    ClinicalReasoningAgent
)

from orchestrator.liver_orchestrator import (
    LiverAIOrchestrator
)


# ============================================================
# CREATE FIBROSIS AGENT
# ============================================================

def create_fibrosis_agent():

    print()
    print("-" * 70)
    print("FIBROSIS AGENT")
    print("-" * 70)

    model = load_pickle_model(
        FIBROSIS_MODEL
    )

    print()
    print("Fibrosis model information:")

    if hasattr(model, "feature_names_in_"):

        print(
            "Features:",
            list(model.feature_names_in_)
        )

    if hasattr(model, "n_features_in_"):

        print(
            "Number of features:",
            model.n_features_in_
        )

    if hasattr(model, "classes_"):

        print(
            "Classes:",
            model.classes_
        )

    agent = FibrosisAgent(
        model=model
    )

    print(
        "✓ Fibrosis Agent initialized"
    )

    return agent


# ============================================================
# CREATE CLINICAL REASONING AGENT
# ============================================================

def create_clinical_reasoning_agent():

    print()
    print("-" * 70)
    print("CLINICAL REASONING AGENT")
    print("-" * 70)

    agent = ClinicalReasoningAgent()

    print(
        "✓ Clinical Reasoning Agent initialized"
    )

    return agent


# ============================================================
# CREATE ALL AGENTS
# ============================================================

def create_agents():

    print()
    print("=" * 80)
    print("INITIALIZING LIVERAI AGENTS")
    print("=" * 80)

    agents = {
        "fatty_liver": None,
        "fibrosis": None,
        "cirrhosis": None,
        "tumor": None,
        "segmentation": None,
        "clinical_reasoning": None
    }

    # --------------------------------------------------------
    # FATTY LIVER
    # --------------------------------------------------------

    print()
    print("[Fatty Liver Agent]")

    print(
        "⚠ No trained Fatty Liver model found."
    )

    print(
        "   Agent requires a trained model."
    )


    # --------------------------------------------------------
    # FIBROSIS
    # --------------------------------------------------------

    try:

        agents["fibrosis"] = (
            create_fibrosis_agent()
        )

    except Exception as e:

        print(
            "✗ Fibrosis Agent failed:"
        )

        print(
            type(e).__name__,
            str(e)
        )

        traceback.print_exc()


    # --------------------------------------------------------
    # CIRRHOSIS
    # --------------------------------------------------------

    print()
    print("[Cirrhosis Agent]")

    print(
        "⚠ No trained Cirrhosis model file found."
    )


    # --------------------------------------------------------
    # TUMOR
    # --------------------------------------------------------

    print()
    print("[Tumor Classification Agent]")

    print(
        "⚠ No trained Tumor model file found."
    )


    # --------------------------------------------------------
    # SEGMENTATION
    # --------------------------------------------------------

    print()
    print("[Liver Segmentation Agent]")

    print(
        "⚠ No trained Segmentation model file found."
    )


    # --------------------------------------------------------
    # CLINICAL REASONING
    # --------------------------------------------------------

    try:

        agents["clinical_reasoning"] = (
            create_clinical_reasoning_agent()
        )

    except Exception as e:

        print(
            "✗ Clinical Reasoning Agent failed:"
        )

        print(
            type(e).__name__,
            str(e)
        )

        traceback.print_exc()


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("INITIALIZATION SUMMARY")
    print("=" * 80)

    for name, agent in agents.items():

        if agent is None:

            print(
                f"✗ {name}"
            )

        else:

            print(
                f"✓ {name}"
            )

    initialized = sum(
        agent is not None
        for agent in agents.values()
    )

    print()
    print(
        f"Initialized: {initialized}/6"
    )

    print("=" * 80)

    return agents


# ============================================================
# CREATE ORCHESTRATOR
# ============================================================

def create_orchestrator():

    agents = create_agents()

    orchestrator = LiverAIOrchestrator(

        fatty_agent=agents["fatty_liver"],

        fibrosis_agent=agents["fibrosis"],

        cirrhosis_agent=agents["cirrhosis"],

        tumor_agent=agents["tumor"],

        segmentation_agent=agents["segmentation"],

        clinical_reasoning_agent=agents[
            "clinical_reasoning"
        ]
    )

    print()
    print("=" * 80)
    print("✓ LIVERAI ORCHESTRATOR CREATED")
    print("=" * 80)

    return orchestrator


# ============================================================
# ARCHITECTURE
# ============================================================

def show_architecture():

    print("""
======================================================================

                    LIVERAI MULTI-AGENT SYSTEM

                              PATIENT
                                 |
                                 v
                     +----------------------+
                     | LIVERAI ORCHESTRATOR |
                     +----------+-----------+
                                |
              +-----------------+-----------------+
              |        |        |        |        |
              v        v        v        v        v
            FAT      FIBROSIS CIRRHOSIS TUMOR  SEGMENTATION
              |        |        |        |        |
              +--------+--------+--------+--------+
                                |
                                v
                    +------------------------+
                    | CLINICAL REASONING      |
                    | AGENT                   |
                    +-----------+------------+
                                |
                                v
                     UNIFIED ASSESSMENT

======================================================================
""")


# ============================================================
# SYSTEM STATUS
# ============================================================

def system_status(orchestrator):

    print()
    print("=" * 80)
    print("SYSTEM STATUS")
    print("=" * 80)

    if hasattr(orchestrator, "agents"):

        for name, agent in orchestrator.agents.items():

            if agent is None:

                print(
                    f"✗ {name:<25} NOT INITIALIZED"
                )

            else:

                print(
                    f"✓ {name:<25} READY"
                )

    else:

        print(
            "⚠ Orchestrator has no 'agents' attribute"
        )

    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("STARTING LIVERAI")
    print("=" * 80)

    check_paths()

    show_model_inventory()

    show_model_paths()

    show_architecture()

    orchestrator = create_orchestrator()

    system_status(
        orchestrator
    )

    return orchestrator
