# =============================================================================
# LiverAI-MultiAgent - main.py
# =============================================================================
# Architecture:
#
#                     LiverAI Orchestrator
#                              |
#          +-------------------+-------------------+
#          |         |         |         |         |
#          v         v         v         v         v
#       Fatty     Fibrosis  Cirrhosis  Tumor   Segmentation
#       Liver       Agent     Agent     Agent      Agent
#          |         |         |         |         |
#          +---------+---------+---------+---------+
#                              |
#                              v
#                  Clinical Reasoning Agent
#                              |
#                              v
#                    Unified Assessment
#
# Google Drive contains datasets/models.
# GitHub contains source code.
# =============================================================================

import os
import sys
import glob
import pickle
import joblib
import traceback
from pathlib import Path

# =============================================================================
# PROJECT PATH
# =============================================================================

PROJECT_ROOT = Path("/content/LiverAI-MultiAgent")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# GOOGLE DRIVE
# =============================================================================

DRIVE_ROOT = Path("/content/drive/MyDrive")

# =============================================================================
# DATASET PATHS
# =============================================================================

PATHS = {

    # -------------------------------------------------------------------------
    # Fatty Liver
    # -------------------------------------------------------------------------
    "fatty_root":
        DRIVE_ROOT / "FattyLiver Agent",

    "fatty_data":
        DRIVE_ROOT / "FattyLiver Agent" / "DATA",

    # -------------------------------------------------------------------------
    # Fibrosis
    # -------------------------------------------------------------------------
    "fibrosis_root":
        DRIVE_ROOT / "Fibrosis Agent",

    "fibrosis_model":
        DRIVE_ROOT /
        "Fibrosis Agent" /
        "XGBoost_model" /
        "xgboost_nafld.pkl",

    # -------------------------------------------------------------------------
    # Cirrhosis
    # -------------------------------------------------------------------------
    "cirrhosis_root":
        DRIVE_ROOT / ".Cirrhosis Agent",

    "cirrhosis_data":
        DRIVE_ROOT /
        ".Cirrhosis Agent" /
        "DATA" /
        "liver_cirrhosis.csv",

    # -------------------------------------------------------------------------
    # Tumor
    # -------------------------------------------------------------------------
    "tumor_root":
        DRIVE_ROOT / "Liver CT Image Dataset",

    # -------------------------------------------------------------------------
    # Segmentation
    # -------------------------------------------------------------------------
    "segmentation_root":
        DRIVE_ROOT / "archive (2)" / "image",

    # -------------------------------------------------------------------------
    # Segmentation alternative
    # -------------------------------------------------------------------------
    "segmentation_root_2":
        DRIVE_ROOT / "image",
}

# =============================================================================
# POSSIBLE TRAINED MODEL EXTENSIONS
# =============================================================================

MODEL_EXTENSIONS = [
    "*.pkl",
    "*.pickle",
    "*.joblib",
    "*.keras",
    "*.h5",
    "*.pth",
    "*.pt",
    "*.onnx",
    "*.safetensors",
]

# =============================================================================
# UTILITY
# =============================================================================

def _exists(path):
    """Return True if a file/folder exists."""
    return Path(path).exists()


# =============================================================================
# CHECK PATHS
# =============================================================================

def check_paths(verbose=True):
    """
    Check all configured Google Drive paths.

    Returns
    -------
    dict
        Dictionary containing paths.
    """

    results = {}

    print("\n" + "=" * 75)
    print("LIVERAI DATASET / MODEL PATH CHECK")
    print("=" * 75)

    for name, path in PATHS.items():

        path = Path(path)
        exists = path.exists()

        results[name] = str(path)

        if exists:
            print(f"✓ {name}")
            print(f"  {path}")

        else:
            print(f"✗ {name}")
            print(f"  {path}")

    print("=" * 75)

    return results


# =============================================================================
# MODEL SEARCH
# =============================================================================

def find_model_files(root=DRIVE_ROOT):
    """
    Search Google Drive recursively for actual trained model files.

    Important:
    .ipynb files are intentionally NOT considered trained models.
    """

    root = Path(root)

    found = []

    for extension in MODEL_EXTENSIONS:

        found.extend(
            root.rglob(extension)
        )

    # Remove duplicates
    found = sorted(
        set(str(p) for p in found)
    )

    return found


# =============================================================================
# SHOW MODEL INVENTORY
# =============================================================================

def show_model_inventory():

    print("\n" + "=" * 75)
    print("LIVERAI MODEL INVENTORY")
    print("=" * 75)

    models = find_model_files()

    if not models:

        print("✗ No trained model files found.")

    else:

        for model in models:

            print(model)

    print("\n")
    print(f"TOTAL TRAINED MODEL FILES: {len(models)}")
    print("=" * 75)

    return models


# =============================================================================
# LOAD PICKLE
# =============================================================================

def load_pickle_model(path):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"Model not found:\n{path}"
        )

    print(f"\nLoading model:")
    print(path)

    try:

        model = joblib.load(path)

    except Exception:

        with open(path, "rb") as f:

            model = pickle.load(f)

    print("✓ Model loaded")
    print(f"  Type: {type(model)}")

    return model


# =============================================================================
# SAFE IMPORTS
# =============================================================================

print("\n" + "=" * 75)
print("IMPORTING LIVERAI COMPONENTS")
print("=" * 75)

# -----------------------------------------------------------------------------
# Agents
# -----------------------------------------------------------------------------

from agents.fatty_liver_agent import FattyLiverAgent
from agents.fibrosis_agent import FibrosisAgent
from agents.cirrhosis_agent import CirrhosisAgent
from agents.tumor_classification_agent import TumorClassificationAgent
from agents.liver_segmentation_agent import LiverSegmentationAgent
from agents.clinical_reasoning_agent import ClinicalReasoningAgent

print("✓ Fatty Liver Agent")
print("✓ Fibrosis Agent")
print("✓ Cirrhosis Agent")
print("✓ Tumor Classification Agent")
print("✓ Liver Segmentation Agent")
print("✓ Clinical Reasoning Agent")

# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------

from orchestrator import LiverAIOrchestrator

print("✓ LiverAI Orchestrator")

# =============================================================================
# CREATE FATY LIVER AGENT
# =============================================================================

def create_fatty_agent(model_path=None):

    print("\n" + "-" * 75)
    print("FATTY LIVER AGENT")
    print("-" * 75)

    if model_path is None:

        print("⚠ No trained Fatty Liver model configured.")
        return None

    if not Path(model_path).exists():

        print(f"⚠ Model not found: {model_path}")
        return None

    model = load_pickle_model(model_path)

    agent = FattyLiverAgent(
        model=model
    )

    print("✓ Fatty Liver Agent initialized")

    return agent


# =============================================================================
# CREATE FIBROSIS AGENT
# =============================================================================

def create_fibrosis_agent():

    print("\n" + "-" * 75)
    print("FIBROSIS AGENT")
    print("-" * 75)

    model_path = PATHS["fibrosis_model"]

    if not model_path.exists():

        print("✗ Fibrosis model not found")
        print(model_path)

        return None

    model = load_pickle_model(model_path)

    # Display model information
    print("\nFibrosis model information:")

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
# CREATE CIRRHOSIS AGENT
# =============================================================================

def create_cirrhosis_agent(model_path=None):

    print("\n" + "-" * 75)
    print("CIRRHOSIS AGENT")
    print("-" * 75)

    if model_path is None:

        print("⚠ No trained Cirrhosis model configured.")

        return None

    if not Path(model_path).exists():

        print(f"⚠ Model not found: {model_path}")

        return None

    package = load_pickle_model(model_path)

    if not isinstance(package, dict):

        print(
            "⚠ Cirrhosis model is not a model package dictionary."
        )

        return None

    agent = CirrhosisAgent(
        model_package=package
    )

    print("✓ Cirrhosis Agent initialized")

    return agent


# =============================================================================
# CREATE TUMOR AGENT
# =============================================================================

def create_tumor_agent(model_path=None):

    print("\n" + "-" * 75)
    print("TUMOR CLASSIFICATION AGENT")
    print("-" * 75)

    if model_path is None:

        print("⚠ No trained Tumor model configured.")

        return None

    model_path = Path(model_path)

    if not model_path.exists():

        print(f"⚠ Tumor model not found:")
        print(model_path)

        return None

    agent = TumorClassificationAgent(
        model_path=str(model_path)
    )

    print("✓ Tumor Classification Agent initialized")

    return agent


# =============================================================================
# CREATE SEGMENTATION AGENT
# =============================================================================

def create_segmentation_agent(model_path=None):

    print("\n" + "-" * 75)
    print("LIVER SEGMENTATION AGENT")
    print("-" * 75)

    if model_path is None:

        print("⚠ No trained Segmentation model configured.")

        return None

    model_path = Path(model_path)

    if not model_path.exists():

        print("⚠ Segmentation model not found:")
        print(model_path)

        return None

    agent = LiverSegmentationAgent(
        model_path=str(model_path)
    )

    print("✓ Liver Segmentation Agent initialized")

    return agent


# =============================================================================
# CREATE CLINICAL REASONING AGENT
# =============================================================================

def create_clinical_reasoning_agent():

    print("\n" + "-" * 75)
    print("CLINICAL REASONING AGENT")
    print("-" * 75)

    agent = ClinicalReasoningAgent()

    print("✓ Clinical Reasoning Agent initialized")

    return agent


# =============================================================================
# AUTOMATIC MODEL DISCOVERY
# =============================================================================

def discover_models():

    """
    Discover trained models in Google Drive.

    This function DOES NOT treat notebooks or datasets as models.
    """

    models = find_model_files()

    inventory = {
        "fatty_liver": [],
        "fibrosis": [],
        "cirrhosis": [],
        "tumor": [],
        "segmentation": [],
    }

    for model in models:

        lower = model.lower()

        if "fatty" in lower or "nafld" in lower:

            inventory["fatty_liver"].append(model)

        elif "fibrosis" in lower:

            inventory["fibrosis"].append(model)

        elif "cirrhosis" in lower:

            inventory["cirrhosis"].append(model)

        elif "tumor" in lower or "mobilenet" in lower or "efficientnet" in lower:

            inventory["tumor"].append(model)

        elif (
            "segment" in lower
            or "unet" in lower
            or "segresnet" in lower
            or "vnet" in lower
        ):

            inventory["segmentation"].append(model)

    return inventory


# =============================================================================
# CREATE ALL AGENTS
# =============================================================================

def create_agents():

    print("\n" + "=" * 75)
    print("INITIALIZING ALL LIVERAI AGENTS")
    print("=" * 75)

    models = discover_models()

    agents = {}

    # -------------------------------------------------------------------------
    # Fatty Liver
    # -------------------------------------------------------------------------

    fatty_agent = None

    if models["fatty_liver"]:

        try:

            fatty_agent = create_fatty_agent(
                models["fatty_liver"][0]
            )

        except Exception as e:

            print(
                f"✗ Fatty Liver Agent failed: {e}"
            )

            traceback.print_exc()

    else:

        print(
            "⚠ Fatty Liver Agent skipped: "
            "no trained model found"
        )

    agents["fatty_liver"] = fatty_agent

    # -------------------------------------------------------------------------
    # Fibrosis
    # -------------------------------------------------------------------------

    fibrosis_agent = None

    try:

        fibrosis_agent = create_fibrosis_agent()

    except Exception as e:

        print(
            f"✗ Fibrosis Agent failed: {e}"
        )

        traceback.print_exc()

    agents["fibrosis"] = fibrosis_agent

    # -------------------------------------------------------------------------
    # Cirrhosis
    # -------------------------------------------------------------------------

    cirrhosis_agent = None

    if models["cirrhosis"]:

        try:

            cirrhosis_agent = create_cirrhosis_agent(
                models["cirrhosis"][0]
            )

        except Exception as e:

            print(
                f"✗ Cirrhosis Agent failed: {e}"
            )

            traceback.print_exc()

    else:

        print(
            "⚠ Cirrhosis Agent skipped: "
            "no trained model found"
        )

    agents["cirrhosis"] = cirrhosis_agent

    # -------------------------------------------------------------------------
    # Tumor
    # -------------------------------------------------------------------------

    tumor_agent = None

    if models["tumor"]:

        try:

            tumor_agent = create_tumor_agent(
                models["tumor"][0]
            )

        except Exception as e:

            print(
                f"✗ Tumor Agent failed: {e}"
            )

            traceback.print_exc()

    else:

        print(
            "⚠ Tumor Classification Agent skipped: "
            "no trained model found"
        )

    agents["tumor"] = tumor_agent

    # -------------------------------------------------------------------------
    # Segmentation
    # -------------------------------------------------------------------------

    segmentation_agent = None

    if models["segmentation"]:

        try:

            segmentation_agent = create_segmentation_agent(
                models["segmentation"][0]
            )

        except Exception as e:

            print(
                f"✗ Segmentation Agent failed: {e}"
            )

            traceback.print_exc()

    else:

        print(
            "⚠ Liver Segmentation Agent skipped: "
            "no trained model found"
        )

    agents["segmentation"] = segmentation_agent

    # -------------------------------------------------------------------------
    # Clinical Reasoning
    # -------------------------------------------------------------------------

    clinical_agent = None

    try:

        clinical_agent = create_clinical_reasoning_agent()

    except Exception as e:

        print(
            f"✗ Clinical Reasoning Agent failed: {e}"
        )

        traceback.print_exc()

    agents["clinical_reasoning"] = clinical_agent

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print("\n" + "=" * 75)
    print("INITIALIZATION SUMMARY")
    print("=" * 75)

    initialized = 0

    for name, agent in agents.items():

        if agent is not None:

            print(f"✓ {name}")

            initialized += 1

        else:

            print(f"✗ {name}")

    print(
        f"\nInitialized: "
        f"{initialized}/{len(agents)}"
    )

    print("=" * 75)

    return agents


# =============================================================================
# CREATE ORCHESTRATOR
# =============================================================================

def create_orchestrator():

    """
    Build the complete LiverAI orchestration layer.
    """

    print("\n" + "=" * 75)
    print("CREATING LIVERAI ORCHESTRATOR")
    print("=" * 75)

    agents = create_agents()

    orchestrator = LiverAIOrchestrator(

        fatty_agent=agents["fatty_liver"],

        fibrosis_agent=agents["fibrosis"],

        cirrhosis_agent=agents["cirrhosis"],

        tumor_agent=agents["tumor"],

        segmentation_agent=agents["segmentation"],

        clinical_reasoning_agent=agents["clinical_reasoning"],
    )

    print("\n✓ LIVERAI ORCHESTRATOR CREATED")

    return orchestrator


# =============================================================================
# ARCHITECTURE
# =============================================================================

def show_architecture():

    print(
        """

                    ┌─────────────────────────┐
                    │   LiverAI Orchestrator   │
                    └────────────┬────────────┘
                                 │
       ┌─────────────┬───────────┼─────────────┬──────────────┐
       ▼             ▼           ▼             ▼              ▼
   Fatty Liver    Fibrosis   Cirrhosis      Tumor       Segmentation
      Agent        Agent       Agent         Agent          Agent
       │             │           │             │              │
       ▼             ▼           ▼             ▼              ▼
   LightGBM       XGBoost     XGBoost     EfficientNet     SegResNet
   / RF           / RF        / RF        / MobileNet      / U-Net
                                                              
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Clinical Reasoning Agent│
                    └────────────┬────────────┘
                                 ▼
                       Unified Assessment

        """
    )


# =============================================================================
# SYSTEM STATUS
# =============================================================================

def system_status(orchestrator):

    print("\n" + "=" * 75)
    print("LIVERAI SYSTEM STATUS")
    print("=" * 75)

    if not hasattr(orchestrator, "agents"):

        print("⚠ Orchestrator has no 'agents' attribute.")

        return

    for name, agent in orchestrator.agents.items():

        if agent is None:

            print(
                f"✗ {name:<25} NOT READY"
            )

        else:

            print(
                f"✓ {name:<25} READY"
            )

    print("=" * 75)


# =============================================================================
# COMPLETE SYSTEM
# =============================================================================

def initialize_liverai():

    print("\n")
    print("=" * 75)
    print("LIVERAI MULTI-AGENT SYSTEM")
    print("=" * 75)

    print(
        f"PROJECT:\n{PROJECT_ROOT}"
    )

    print(
        f"GOOGLE DRIVE:\n{DRIVE_ROOT}"
    )

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------

    print("\n" + "=" * 75)
    print("CHECKING PATHS")
    print("=" * 75)

    check_paths()

    # -------------------------------------------------------------------------
    # Model inventory
    # -------------------------------------------------------------------------

    show_model_inventory()

    # -------------------------------------------------------------------------
    # Architecture
    # -------------------------------------------------------------------------

    show_architecture()

    # -------------------------------------------------------------------------
    # Orchestrator
    # -------------------------------------------------------------------------

    orchestrator = create_orchestrator()

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    system_status(orchestrator)

    return orchestrator


# =============================================================================
# MAIN
# =============================================================================

def main():

    return initialize_liverai()


# =============================================================================
# DIRECT EXECUTION
# =============================================================================

if __name__ == "__main__":

    orchestrator = main()
