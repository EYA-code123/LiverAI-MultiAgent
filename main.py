# ============================================================
# WRITE CORRECTED main.py
# ============================================================

import os
import sys
from pathlib import Path

PROJECT_DIR = "/content/LiverAI-MultiAgent"

MAIN_FILE = os.path.join(PROJECT_DIR, "main.py")

os.makedirs(PROJECT_DIR, exist_ok=True)

main_code = r'''
# =============================================================================
# LiverAI-MultiAgent
# MAIN ENTRY POINT
# =============================================================================

import os
import sys
import pickle
import joblib
import traceback


# =============================================================================
# PROJECT
# =============================================================================

PROJECT_DIR = "/content/LiverAI-MultiAgent"

DRIVE_BASE = "/content/drive/MyDrive"


# =============================================================================
# REAL DATA / MODEL LOCATIONS
# =============================================================================

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

CIRRHOSIS_MODEL = os.path.join(
    CIRRHOSIS_ROOT,
    "models",
    "XGBoost_Cirrhosis.pkl"
)


TUMOR_ROOT = os.path.join(
    DRIVE_BASE,
    "Liver CT Image Dataset"
)

TUMOR_MODEL = os.path.join(
    TUMOR_ROOT,
    "model.keras"
)


SEGMENTATION_ROOT = os.path.join(
    DRIVE_BASE,
    "archive (2)"
)

SEGMENTATION_MODEL = os.path.join(
    SEGMENTATION_ROOT,
    "segresnet.pth"
)


# =============================================================================
# PYTHON PATH
# =============================================================================

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


os.chdir(PROJECT_DIR)


# =============================================================================
# IMPORT AGENTS
# =============================================================================

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


# =============================================================================
# HEADER
# =============================================================================

print("=" * 80)
print("LIVERAI MULTI-AGENT SYSTEM")
print("=" * 80)

print("PROJECT:")
print(PROJECT_DIR)

print("GOOGLE DRIVE:")
print(DRIVE_BASE)

print("=" * 80)


# =============================================================================
# CHECK PATHS
# =============================================================================

def check_paths():

    print()
    print("=" * 80)
    print("LIVERAI DATASET / MODEL PATH CHECK")
    print("=" * 80)

    paths = {

        "Fatty Liver root":
            FATTY_ROOT,

        "Fatty Liver data":
            FATTY_DATA,

        "Fibrosis root":
            FIBROSIS_ROOT,

        "Fibrosis model":
            FIBROSIS_MODEL,

        "Cirrhosis root":
            CIRRHOSIS_ROOT,

        "Cirrhosis data":
            CIRRHOSIS_DATA,

        "Cirrhosis model":
            CIRRHOSIS_MODEL,

        "Tumor root":
            TUMOR_ROOT,

        "Tumor model":
            TUMOR_MODEL,

        "Segmentation root":
            SEGMENTATION_ROOT,

        "Segmentation model":
            SEGMENTATION_MODEL
    }

    results = {}

    for name, path in paths.items():

        exists = os.path.exists(path)

        results[name] = path

        if exists:
            print(f"✓ {name}")
            print(f"  {path}")
        else:
            print(f"✗ {name}")
            print(f"  {path}")

    print("=" * 80)

    return results


# =============================================================================
# FIND MODEL FILES
# =============================================================================

def find_model_files():

    print()
    print("=" * 80)
    print("SEARCHING FOR TRAINED MODEL FILES")
    print("=" * 80)

    extensions = (
        ".pkl",
        ".pickle",
        ".joblib",
        ".keras",
        ".h5",
        ".pth",
        ".pt"
    )

    roots = [
        FATTY_ROOT,
        FIBROSIS_ROOT,
        CIRRHOSIS_ROOT,
        TUMOR_ROOT,
        SEGMENTATION_ROOT
    ]

    found = []

    for root in roots:

        if not os.path.exists(root):
            continue

        for current_root, dirs, files in os.walk(root):

            for filename in files:

                if filename.lower().endswith(extensions):

                    path = os.path.join(
                        current_root,
                        filename
                    )

                    found.append(path)
                    print(path)

    print()
    print("=" * 80)
    print("TOTAL MODEL FILES FOUND:", len(found))
    print("=" * 80)

    return found


# =============================================================================
# MODEL INVENTORY
# =============================================================================

def show_model_inventory():

    print()
    print("=" * 80)
    print("LIVERAI MODEL INVENTORY")
    print("=" * 80)

    inventory = {

        "Fatty Liver":
            FATTY_ROOT,

        "Fibrosis":
            FIBROSIS_ROOT,

        "Cirrhosis":
            CIRRHOSIS_ROOT,

        "Tumor":
            TUMOR_ROOT,

        "Segmentation":
            SEGMENTATION_ROOT
    }

    for name, root in inventory.items():

        print()
        print("-" * 60)
        print(name)
        print("-" * 60)

        if not os.path.exists(root):

            print("Root not found")
            continue

        found = False

        for current_root, dirs, files in os.walk(root):

            for filename in files:

                if filename.lower().endswith(
                    (
                        ".pkl",
                        ".pickle",
                        ".joblib",
                        ".keras",
                        ".h5",
                        ".pth",
                        ".pt"
                    )
                ):

                    print(
                        os.path.join(
                            current_root,
                            filename
                        )
                    )

                    found = True

        if not found:
            print("No trained model found")


# =============================================================================
# MODEL PATHS
# =============================================================================

def show_model_paths():

    print()
    print("=" * 80)
    print("MODEL PATHS")
    print("=" * 80)

    paths = {

        "Fatty Liver":
            "NOT AVAILABLE",

        "Fibrosis":
            FIBROSIS_MODEL,

        "Cirrhosis":
            CIRRHOSIS_MODEL,

        "Tumor":
            TUMOR_MODEL,

        "Segmentation":
            SEGMENTATION_MODEL
    }

    for name, path in paths.items():

        if path == "NOT AVAILABLE":

            print(
                f"⚠ {name:<20}: NO TRAINED MODEL"
            )

        elif os.path.exists(path):

            print(
                f"✓ {name:<20}: {path}"
            )

        else:

            print(
                f"✗ {name:<20}: NOT FOUND"
            )

    print("=" * 80)


# =============================================================================
# LOAD JOBLIB / PICKLE
# =============================================================================

def load_pickle_model(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Model not found: {path}"
        )

    try:

        model = joblib.load(path)

    except Exception:

        with open(path, "rb") as f:

            model = pickle.load(f)

    return model


# =============================================================================
# FAT MODEL
# =============================================================================

def create_fatty_agent():

    print()
    print("-" * 70)
    print("FATTY LIVER AGENT")
    print("-" * 70)

    raise FileNotFoundError(
        "No trained Fatty Liver model is currently available. "
        "Train and save one before initializing this agent."
    )


# =============================================================================
# FIBROSIS
# =============================================================================

def create_fibrosis_agent():

    print()
    print("-" * 70)
    print("FIBROSIS AGENT")
    print("-" * 70)

    model = load_pickle_model(
        FIBROSIS_MODEL
    )

    print("✓ Model loaded")
    print("Type:", type(model))

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

    print("✓ Fibrosis Agent initialized")

    return agent


# =============================================================================
# CIRRHOSIS
# =============================================================================

def create_cirrhosis_agent():

    print()
    print("-" * 70)
    print("CIRRHOSIS AGENT")
    print("-" * 70)

    if not os.path.exists(CIRRHOSIS_MODEL):

        raise FileNotFoundError(
            "Cirrhosis model not found.\n"
            "Expected:\n"
            + CIRRHOSIS_MODEL
        )

    package = load_pickle_model(
        CIRRHOSIS_MODEL
    )

    if not isinstance(package, dict):

        raise TypeError(
            "Cirrhosis artifact must be a dictionary."
        )

    print(
        "Package keys:",
        list(package.keys())
    )

    agent = CirrhosisAgent(
        model_package=package
    )

    print("✓ Cirrhosis Agent initialized")

    return agent


# =============================================================================
# TUMOR
# =============================================================================

def create_tumor_agent():

    print()
    print("-" * 70)
    print("TUMOR CLASSIFICATION AGENT")
    print("-" * 70)

    if not os.path.exists(TUMOR_MODEL):

        raise FileNotFoundError(
            "Tumor model not found.\n"
            "Expected:\n"
            + TUMOR_MODEL
        )

    agent = TumorClassificationAgent(
        model_path=TUMOR_MODEL
    )

    print("✓ Tumor Classification Agent initialized")

    return agent


# =============================================================================
# SEGMENTATION
# =============================================================================

def create_segmentation_agent():

    print()
    print("-" * 70)
    print("LIVER SEGMENTATION AGENT")
    print("-" * 70)

    if not os.path.exists(SEGMENTATION_MODEL):

        raise FileNotFoundError(
            "Segmentation model not found.\n"
            "Expected:\n"
            + SEGMENTATION_MODEL
        )

    agent = LiverSegmentationAgent(
        model_path=SEGMENTATION_MODEL
    )

    print("✓ Liver Segmentation Agent initialized")

    return agent


# =============================================================================
# CLINICAL REASONING
# =============================================================================

def create_clinical_reasoning_agent():

    print()
    print("-" * 70)
    print("CLINICAL REASONING AGENT")
    print("-" * 70)

    agent = ClinicalReasoningAgent()

    print("✓ Clinical Reasoning Agent initialized")

    return agent


# =============================================================================
# CREATE ALL AGENTS
# =============================================================================

def create_agents():

    print()
    print("=" * 80)
    print("INITIALIZING ALL AGENTS")
    print("=" * 80)

    agents = {

        "fatty_liver": None,
        "fibrosis": None,
        "cirrhosis": None,
        "tumor": None,
        "segmentation": None,
        "clinical_reasoning": None
    }


    constructors = {

        "fatty_liver":
            create_fatty_agent,

        "fibrosis":
            create_fibrosis_agent,

        "cirrhosis":
            create_cirrhosis_agent,

        "tumor":
            create_tumor_agent,

        "segmentation":
            create_segmentation_agent,

        "clinical_reasoning":
            create_clinical_reasoning_agent
    }


    for name, constructor in constructors.items():

        try:

            agents[name] = constructor()

        except Exception as e:

            print(
                f"✗ {name} skipped:"
            )

            print(
                f"  {type(e).__name__}: {e}"
            )


    print()
    print("=" * 80)
    print("INITIALIZATION SUMMARY")
    print("=" * 80)

    initialized = 0

    for name, agent in agents.items():

        if agent is not None:

            print(f"✓ {name}")
            initialized += 1

        else:

            print(f"✗ {name}")


    print()
    print(
        f"Initialized: {initialized}/{len(agents)}"
    )

    print("=" * 80)

    return agents


# =============================================================================
# CREATE ORCHESTRATOR
# =============================================================================

def create_orchestrator():

    agents = create_agents()

    orchestrator = LiverAIOrchestrator(

        fatty_agent=
            agents["fatty_liver"],

        fibrosis_agent=
            agents["fibrosis"],

        cirrhosis_agent=
            agents["cirrhosis"],

        tumor_agent=
            agents["tumor"],

        segmentation_agent=
            agents["segmentation"],

        clinical_reasoning_agent=
            agents["clinical_reasoning"]
    )

    print()
    print("=" * 80)
    print("✓ LIVERAI ORCHESTRATOR CREATED")
    print("=" * 80)

    return orchestrator


# =============================================================================
# ARCHITECTURE
# =============================================================================

def show_architecture():

    print(
r"""
================================================================
                    LIVERAI MULTI-AGENT
================================================================

                         PATIENT DATA
                              |
                              v
                   +---------------------+
                   | LIVERAI ORCHESTRATOR|
                   +----------+----------+
                              |
        +----------+----------+----------+----------+
        |          |          |          |          |
        v          v          v          v          v
      FAT/NAFLD FIBROSIS  CIRRHOSIS   TUMOR   SEGMENTATION
        |          |          |          |          |
        +----------+----------+----------+----------+
                              |
                              v
                 +-------------------------+
                 | CLINICAL REASONING      |
                 | AGENT                   |
                 +-----------+-------------+
                             |
                             v
                  UNIFIED LIVER ASSESSMENT

================================================================
"""
    )


# =============================================================================
# SYSTEM STATUS
# =============================================================================

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

    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():

    check_paths()

    show_model_inventory()

    show_architecture()

    orchestrator = create_orchestrator()

    system_status(
        orchestrator
    )

    return orchestrator


# =============================================================================
# END
# =============================================================================
'''

with open(MAIN_FILE, "w", encoding="utf-8") as f:
    f.write(main_code)

print("✓ main.py rewritten")
print(MAIN_FILE)
