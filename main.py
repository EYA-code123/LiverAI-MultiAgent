# =============================================================================
# LiverAI-MultiAgent
# MAIN ENTRY POINT
# ROBUST MULTI-AGENT INITIALIZATION
# =============================================================================

import os
import sys
import pickle
import joblib
import traceback
from pathlib import Path
from datetime import datetime


# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================

PROJECT_DIR = "/content/LiverAI-MultiAgent"
DRIVE_BASE = "/content/drive/MyDrive"

# Add project to Python path
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# Change working directory
if os.path.exists(PROJECT_DIR):
    os.chdir(PROJECT_DIR)


# =============================================================================
# REAL GOOGLE DRIVE DIRECTORIES
# =============================================================================

FATTY_ROOT = os.path.join(
    DRIVE_BASE,
    "FattyLiver Agent"
)

FIBROSIS_ROOT = os.path.join(
    DRIVE_BASE,
    "Fibrosis Agent"
)

CIRRHOSIS_ROOT = os.path.join(
    DRIVE_BASE,
    ".Cirrhosis Agent"
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


# =============================================================================
# REAL DATA PATHS
# =============================================================================

FATTY_DATA_DIR = os.path.join(
    FATTY_ROOT,
    "DATA"
)

CIRRHOSIS_DATA_DIR = os.path.join(
    CIRRHOSIS_ROOT,
    "DATA"
)

CIRRHOSIS_DATA_PATH = os.path.join(
    CIRRHOSIS_DATA_DIR,
    "liver_cirrhosis.csv"
)


# =============================================================================
# REAL TRAINED MODEL PATHS
# =============================================================================

# Fibrosis model - confirmed existing
FIBROSIS_MODEL_PATH = os.path.join(
    FIBROSIS_ROOT,
    "XGBoost_model",
    "xgboost_nafld.pkl"
)


# =============================================================================
# POSSIBLE MODEL DIRECTORIES
# =============================================================================

FATTY_MODEL_DIRS = [
    os.path.join(FATTY_ROOT, "models"),
    os.path.join(FATTY_ROOT, "MODEL"),
    os.path.join(FATTY_ROOT, "Models")
]

CIRRHOSIS_MODEL_DIRS = [
    os.path.join(CIRRHOSIS_ROOT, "models"),
    os.path.join(CIRRHOSIS_ROOT, "MODEL"),
    os.path.join(CIRRHOSIS_ROOT, "Models")
]

TUMOR_MODEL_DIRS = [
    os.path.join(TUMOR_ROOT, "models"),
    os.path.join(TUMOR_ROOT, "MODEL"),
    os.path.join(TUMOR_ROOT, "Models")
]

SEGMENTATION_MODEL_DIRS = [
    os.path.join(SEGMENTATION_ROOT, "models"),
    os.path.join(SEGMENTATION_ROOT, "MODEL"),
    os.path.join(SEGMENTATION_ROOT, "Models")
]


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
# IMPORT AGENTS
# =============================================================================

def import_agents():

    print("\n")
    print("=" * 80)
    print("IMPORTING AGENTS")
    print("=" * 80)

    agents = {}

    # -------------------------------------------------------------------------
    # FAT
    # -------------------------------------------------------------------------

    try:
        from agents.fatty_liver_agent import FattyLiverAgent

        agents["fatty_liver"] = FattyLiverAgent

        print("✓ Fatty Liver Agent")

    except Exception as e:

        agents["fatty_liver"] = None

        print("✗ Fatty Liver Agent")
        print("  Error:", repr(e))

    # -------------------------------------------------------------------------
    # FIBROSIS
    # -------------------------------------------------------------------------

    try:
        from agents.fibrosis_agent import FibrosisAgent

        agents["fibrosis"] = FibrosisAgent

        print("✓ Fibrosis Agent")

    except Exception as e:

        agents["fibrosis"] = None

        print("✗ Fibrosis Agent")
        print("  Error:", repr(e))

    # -------------------------------------------------------------------------
    # CIRRHOSIS
    # -------------------------------------------------------------------------

    try:
        from agents.cirrhosis_agent import CirrhosisAgent

        agents["cirrhosis"] = CirrhosisAgent

        print("✓ Cirrhosis Agent")

    except Exception as e:

        agents["cirrhosis"] = None

        print("✗ Cirrhosis Agent")
        print("  Error:", repr(e))

    # -------------------------------------------------------------------------
    # TUMOR
    # -------------------------------------------------------------------------

    try:
        from agents.tumor_classification_agent import (
            TumorClassificationAgent
        )

        agents["tumor"] = TumorClassificationAgent

        print("✓ Tumor Classification Agent")

    except Exception as e:

        agents["tumor"] = None

        print("✗ Tumor Classification Agent")
        print("  Error:", repr(e))

    # -------------------------------------------------------------------------
    # SEGMENTATION
    # -------------------------------------------------------------------------

    try:
        from agents.liver_segmentation_agent import (
            LiverSegmentationAgent
        )

        agents["segmentation"] = LiverSegmentationAgent

        print("✓ Liver Segmentation Agent")

    except Exception as e:

        agents["segmentation"] = None

        print("✗ Liver Segmentation Agent")
        print("  Error:", repr(e))

    # -------------------------------------------------------------------------
    # CLINICAL
    # -------------------------------------------------------------------------

    try:
        from agents.clinical_reasoning_agent import (
            ClinicalReasoningAgent
        )

        agents["clinical_reasoning"] = ClinicalReasoningAgent

        print("✓ Clinical Reasoning Agent")

    except Exception as e:

        agents["clinical_reasoning"] = None

        print("✗ Clinical Reasoning Agent")
        print("  Error:", repr(e))

    print("=" * 80)

    return agents


# Import classes
AGENT_CLASSES = import_agents()


# =============================================================================
# CHECK PATHS
# =============================================================================

def check_path(name, path):

    exists = os.path.exists(path)

    if exists:

        print(f"✓ {name}")
        print(f"    {path}")

    else:

        print(f"✗ {name}")
        print(f"    {path}")

    return exists


def check_paths():

    print("\n")
    print("=" * 80)
    print("LIVERAI DATASET / MODEL PATH CHECK")
    print("=" * 80)

    results = {}

    results["fatty_root"] = check_path(
        "Fatty Liver root",
        FATTY_ROOT
    )

    results["fatty_data"] = check_path(
        "Fatty Liver data",
        FATTY_DATA_DIR
    )

    results["fibrosis_root"] = check_path(
        "Fibrosis root",
        FIBROSIS_ROOT
    )

    results["fibrosis_model"] = check_path(
        "Fibrosis model",
        FIBROSIS_MODEL_PATH
    )

    results["cirrhosis_root"] = check_path(
        "Cirrhosis root",
        CIRRHOSIS_ROOT
    )

    results["cirrhosis_data"] = check_path(
        "Cirrhosis data",
        CIRRHOSIS_DATA_PATH
    )

    results["tumor_root"] = check_path(
        "Tumor root",
        TUMOR_ROOT
    )

    results["segmentation_root"] = check_path(
        "Segmentation root",
        SEGMENTATION_ROOT
    )

    print("=" * 80)

    return results


# =============================================================================
# MODEL SEARCH
# =============================================================================

MODEL_EXTENSIONS = (
    ".pkl",
    ".pickle",
    ".joblib",
    ".pth",
    ".pt",
    ".keras",
    ".h5",
    ".safetensors"
)


def search_models(root):

    found = []

    if not os.path.exists(root):
        return found

    for current_root, dirs, files in os.walk(root):

        # Ignore unnecessary directories
        dirs[:] = [
            d for d in dirs
            if d not in [
                ".git",
                "__pycache__",
                ".ipynb_checkpoints"
            ]
        ]

        for file in files:

            lower = file.lower()

            if lower.endswith(MODEL_EXTENSIONS):

                full_path = os.path.join(
                    current_root,
                    file
                )

                found.append(full_path)

    return sorted(found)


# =============================================================================
# SHOW MODEL INVENTORY
# =============================================================================

def show_model_inventory():

    print("\n")
    print("=" * 80)
    print("SEARCHING FOR TRAINED MODEL FILES")
    print("=" * 80)

    all_models = []

    search_roots = {
        "Fatty Liver": FATTY_ROOT,
        "Fibrosis": FIBROSIS_ROOT,
        "Cirrhosis": CIRRHOSIS_ROOT,
        "Tumor": TUMOR_ROOT,
        "Segmentation": SEGMENTATION_ROOT
    }

    for agent_name, root in search_roots.items():

        models = search_models(root)

        print("\n")
        print(f"[{agent_name}]")

        if not models:

            print("  No trained model file found.")

        else:

            for model_path in models:

                print(
                    "  ✓",
                    model_path
                )

                all_models.append(
                    model_path
                )

    print("\n")
    print("=" * 80)
    print(
        f"TOTAL MODEL FILES: {len(all_models)}"
    )
    print("=" * 80)

    return all_models


# =============================================================================
# FIND BEST MODEL
# =============================================================================

def find_model(root, keywords=None):

    models = search_models(root)

    if not models:
        return None

    if keywords:

        for model in models:

            name = os.path.basename(
                model
            ).lower()

            if all(
                keyword.lower() in name
                for keyword in keywords
            ):

                return model

    return models[0]


# =============================================================================
# LOAD PICKLE / JOBLIB
# =============================================================================

def load_pickle_model(path):

    if path is None:
        raise FileNotFoundError(
            "Model path is None."
        )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Model not found:\n{path}"
        )

    print("\nLoading model:")
    print(path)

    # First try joblib
    try:

        model = joblib.load(path)

        print("✓ Model loaded")
        print("  Type:", type(model))

        return model

    except Exception as joblib_error:

        print(
            "⚠ joblib loading failed:"
        )

        print(
            " ",
            repr(joblib_error)
        )

    # Try pickle
    try:

        with open(
            path,
            "rb"
        ) as f:

            model = pickle.load(f)

        print("✓ Pickle model loaded")
        print("  Type:", type(model))

        return model

    except Exception as pickle_error:

        raise RuntimeError(
            f"""
Unable to load model.

Path:
{path}

joblib error:
{joblib_error}

pickle error:
{pickle_error}
"""
        )


# =============================================================================
# LOAD TORCH MODEL PATH
# =============================================================================

def validate_torch_model(path):

    if path is None:

        return False

    if not os.path.exists(path):

        return False

    valid_extensions = (
        ".pth",
        ".pt"
    )

    return path.lower().endswith(
        valid_extensions
    )


# =============================================================================
# FAT IMAGE MODEL PATH
# =============================================================================

def find_fatty_model():

    # Prefer known names
    preferred_names = [
        "fatty_liver.pkl",
        "fatty_liver.joblib",
        "random_forest.pkl",
        "random_forest.joblib",
        "lightgbm.pkl",
        "lightgbm.joblib",
        "decision_tree.pkl",
        "decision_tree.joblib",
        "model.pkl"
    ]

    models = search_models(
        FATTY_ROOT
    )

    for preferred in preferred_names:

        for model in models:

            if os.path.basename(
                model
            ).lower() == preferred.lower():

                return model

    # Otherwise return first compatible pickle
    for model in models:

        if model.lower().endswith(
            (".pkl", ".pickle", ".joblib")
        ):

            return model

    return None


# =============================================================================
# CIRRHOSIS MODEL PATH
# =============================================================================

def find_cirrhosis_model():

    preferred_names = [
        "XGBoost_Cirrhosis.pkl",
        "xgboost_cirrhosis.pkl",
        "cirrhosis.pkl",
        "model.pkl"
    ]

    models = search_models(
        CIRRHOSIS_ROOT
    )

    for preferred in preferred_names:

        for model in models:

            if os.path.basename(
                model
            ).lower() == preferred.lower():

                return model

    for model in models:

        if model.lower().endswith(
            (".pkl", ".pickle", ".joblib")
        ):

            return model

    return None


# =============================================================================
# TUMOR MODEL PATH
# =============================================================================

def find_tumor_model():

    models = search_models(
        TUMOR_ROOT
    )

    # Prefer PyTorch models
    for model in models:

        if model.lower().endswith(
            (".pth", ".pt")
        ):

            return model

    # Then Keras
    for model in models:

        if model.lower().endswith(
            (".keras", ".h5")
        ):

            return model

    return None


# =============================================================================
# SEGMENTATION MODEL PATH
# =============================================================================

def find_segmentation_model():

    # Search the complete Drive project area first
    candidate_roots = [
        SEGMENTATION_ROOT,
        os.path.join(
            DRIVE_BASE,
            "Segmentation Agent"
        ),
        os.path.join(
            DRIVE_BASE,
            "Liver Segmentation Agent"
        ),
        os.path.join(
            DRIVE_BASE,
            "LiverAI"
        )
    ]

    all_models = []

    for root in candidate_roots:

        models = search_models(root)

        for model in models:

            if model not in all_models:

                all_models.append(
                    model
                )

    # Prefer explicit segmentation names
    for model in all_models:

        name = os.path.basename(
            model
        ).lower()

        if (
            "seg" in name
            or "unet" in name
            or "vnet" in name
            or "resnet" in name
        ):

            return model

    # Otherwise PyTorch
    for model in all_models:

        if model.lower().endswith(
            (".pth", ".pt")
        ):

            return model

    return None


# =============================================================================
# CREATE FAT AGENT
# =============================================================================

def create_fatty_agent():

    AgentClass = AGENT_CLASSES.get(
        "fatty_liver"
    )

    if AgentClass is None:

        print(
            "⚠ Fatty Liver Agent class unavailable."
        )

        return None

    model_path = find_fatty_model()

    if model_path is None:

        print(
            "⚠ Fatty Liver Agent skipped:"
            " no trained model found"
        )

        print(
            "  Current folder contains notebooks/data,"
            " but no .pkl/.joblib model."
        )

        return None

    print("\n")
    print("-" * 80)
    print("FATTY LIVER AGENT")
    print("-" * 80)

    model = load_pickle_model(
        model_path
    )

    agent = AgentClass(
        model=model
    )

    print(
        "✓ Fatty Liver Agent initialized"
    )

    return agent


# =============================================================================
# CREATE FIBROSIS AGENT
# =============================================================================

def create_fibrosis_agent():

    AgentClass = AGENT_CLASSES.get(
        "fibrosis"
    )

    if AgentClass is None:

        print(
            "⚠ Fibrosis Agent class unavailable."
        )

        return None

    if not os.path.exists(
        FIBROSIS_MODEL_PATH
    ):

        print(
            "⚠ Fibrosis Agent skipped:"
            " model not found"
        )

        return None

    print("\n")
    print("-" * 80)
    print("FIBROSIS AGENT")
    print("-" * 80)

    model = load_pickle_model(
        FIBROSIS_MODEL_PATH
    )

    # Display model information
    print("\nFibrosis model information:")

    if hasattr(
        model,
        "feature_names_in_"
    ):

        print(
            "  FEATURES:",
            list(model.feature_names_in_)
        )

    if hasattr(
        model,
        "n_features_in_"
    ):

        print(
            "  N FEATURES:",
            model.n_features_in_
        )

    if hasattr(
        model,
        "classes_"
    ):

        print(
            "  CLASSES:",
            model.classes_
        )

    print(
        "  MODEL TYPE:",
        type(model)
    )

    agent = AgentClass(
        model=model
    )

    print(
        "✓ Fibrosis Agent initialized"
    )

    return agent


# =============================================================================
# CREATE CIRRHOSIS AGENT
# =============================================================================

def create_cirrhosis_agent():

    AgentClass = AGENT_CLASSES.get(
        "cirrhosis"
    )

    if AgentClass is None:

        print(
            "⚠ Cirrhosis Agent class unavailable."
        )

        return None

    model_path = find_cirrhosis_model()

    if model_path is None:

        print(
            "⚠ Cirrhosis Agent skipped:"
            " no trained model found"
        )

        print(
            "  Found notebooks/data, but no trained"
            " pickle/joblib model."
        )

        return None

    print("\n")
    print("-" * 80)
    print("CIRRHOSIS AGENT")
    print("-" * 80)

    package = load_pickle_model(
        model_path
    )

    print(
        "Cirrhosis package type:",
        type(package)
    )

    if not isinstance(
        package,
        dict
    ):

        raise TypeError(
            "Cirrhosis model package must be a dictionary."
        )

    print(
        "Cirrhosis package keys:"
    )

    print(
        list(package.keys())
    )

    agent = AgentClass(
        model_package=package
    )

    print(
        "✓ Cirrhosis Agent initialized"
    )

    return agent


# =============================================================================
# CREATE TUMOR AGENT
# =============================================================================

def create_tumor_agent():

    AgentClass = AGENT_CLASSES.get(
        "tumor"
    )

    if AgentClass is None:

        print(
            "⚠ Tumor Agent class unavailable."
        )

        return None

    model_path = find_tumor_model()

    if model_path is None:

        print(
            "⚠ Tumor Classification Agent skipped:"
            " no trained model found"
        )

        print(
            "  Dataset exists, but a trained"
            " .pth/.pt/.keras/.h5 model is required."
        )

        return None

    print("\n")
    print("-" * 80)
    print("TUMOR CLASSIFICATION AGENT")
    print("-" * 80)

    print(
        "Model:",
        model_path
    )

    agent = AgentClass(
        model_path=model_path
    )

    print(
        "✓ Tumor Classification Agent initialized"
    )

    return agent


# =============================================================================
# CREATE SEGMENTATION AGENT
# =============================================================================

def create_segmentation_agent():

    AgentClass = AGENT_CLASSES.get(
        "segmentation"
    )

    if AgentClass is None:

        print(
            "⚠ Segmentation Agent class unavailable."
        )

        return None

    model_path = find_segmentation_model()

    if model_path is None:

        print(
            "⚠ Liver Segmentation Agent skipped:"
            " no trained model found"
        )

        print(
            "  Dataset exists, but a trained"
            " segmentation model is required."
        )

        return None

    print("\n")
    print("-" * 80)
    print("LIVER SEGMENTATION AGENT")
    print("-" * 80)

    print(
        "Model:",
        model_path
    )

    agent = AgentClass(
        model_path=model_path
    )

    print(
        "✓ Liver Segmentation Agent initialized"
    )

    return agent


# =============================================================================
# CREATE CLINICAL REASONING AGENT
# =============================================================================

def create_clinical_reasoning_agent():

    AgentClass = AGENT_CLASSES.get(
        "clinical_reasoning"
    )

    if AgentClass is None:

        print(
            "⚠ Clinical Reasoning Agent unavailable."
        )

        return None

    print("\n")
    print("-" * 80)
    print("CLINICAL REASONING AGENT")
    print("-" * 80)

    agent = AgentClass()

    print(
        "✓ Clinical Reasoning Agent initialized"
    )

    return agent


# =============================================================================
# CREATE ALL AGENTS
# =============================================================================

def create_agents():

    print("\n")
    print("=" * 80)
    print("INITIALIZING ALL LIVERAI AGENTS")
    print("=" * 80)

    agents = {
        "fatty_liver": None,
        "fibrosis": None,
        "cirrhosis": None,
        "tumor": None,
        "segmentation": None,
        "clinical_reasoning": None
    }

    # -------------------------------------------------------------------------
    # FAT
    # -------------------------------------------------------------------------

    try:

        agents["fatty_liver"] = (
            create_fatty_agent()
        )

    except Exception as e:

        print(
            "✗ Fatty Liver Agent failed:"
        )

        print(
            " ",
            repr(e)
        )

        traceback.print_exc()

    # -------------------------------------------------------------------------
    # FIBROSIS
    # -------------------------------------------------------------------------

    try:

        agents["fibrosis"] = (
            create_fibrosis_agent()
        )

    except Exception as e:

        print(
            "✗ Fibrosis Agent failed:"
        )

        print(
            " ",
            repr(e)
        )

        traceback.print_exc()

    # -------------------------------------------------------------------------
    # CIRRHOSIS
    # -------------------------------------------------------------------------

    try:

        agents["cirrhosis"] = (
            create_cirrhosis_agent()
        )

    except Exception as e:

        print(
            "✗ Cirrhosis Agent failed:"
        )

        print(
            " ",
            repr(e)
        )

        traceback.print_exc()

    # -------------------------------------------------------------------------
    # TUMOR
    # -------------------------------------------------------------------------

    try:

        agents["tumor"] = (
            create_tumor_agent()
        )

    except Exception as e:

        print(
            "✗ Tumor Agent failed:"
        )

        print(
            " ",
            repr(e)
        )

        traceback.print_exc()

    # -------------------------------------------------------------------------
    # SEGMENTATION
    # -------------------------------------------------------------------------

    try:

        agents["segmentation"] = (
            create_segmentation_agent()
        )

    except Exception as e:

        print(
            "✗ Segmentation Agent failed:"
        )

        print(
            " ",
            repr(e)
        )

        traceback.print_exc()

    # -------------------------------------------------------------------------
    # CLINICAL
    # -------------------------------------------------------------------------

    try:

        agents["clinical_reasoning"] = (
            create_clinical_reasoning_agent()
        )

    except Exception as e:

        print(
            "✗ Clinical Reasoning Agent failed:"
        )

        print(
            " ",
            repr(e)
        )

        traceback.print_exc()

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("INITIALIZATION SUMMARY")
    print("=" * 80)

    initialized = 0

    for name, agent in agents.items():

        if agent is not None:

            print(
                f"✓ {name:<25} READY"
            )

            initialized += 1

        else:

            print(
                f"✗ {name:<25} NOT READY"
            )

    print("=" * 80)

    print(
        f"Initialized: {initialized}/{len(agents)}"
    )

    print("=" * 80)

    return agents


# =============================================================================
# CREATE ORCHESTRATOR
# =============================================================================

def create_orchestrator():

    print("\n")
    print("=" * 80)
    print("CREATING LIVERAI ORCHESTRATOR")
    print("=" * 80)

    try:

        from orchestrator.liver_orchestrator import (
            LiverAIOrchestrator
        )

    except Exception as e:

        print(
            "✗ Could not import LiverAIOrchestrator"
        )

        print(
            "Error:",
            repr(e)
        )

        traceback.print_exc()

        return None

    agents = create_agents()

    # -------------------------------------------------------------------------
    # CREATE ORCHESTRATOR
    # -------------------------------------------------------------------------

    try:

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

    except TypeError as e:

        print("\n")
        print(
            "✗ Orchestrator constructor mismatch."
        )

        print(
            "Error:",
            repr(e)
        )

        print(
            "\nThe constructor of"
            " LiverAIOrchestrator must accept:"
        )

        print(
            """
fatty_agent
fibrosis_agent
cirrhosis_agent
tumor_agent
segmentation_agent
clinical_reasoning_agent
"""
        )

        raise

    print("\n")
    print("=" * 80)
    print("✓ LIVERAI ORCHESTRATOR CREATED")
    print("=" * 80)

    return orchestrator


# =============================================================================
# ARCHITECTURE
# =============================================================================

def show_architecture():

    print(
        """

===============================================================================
                         LIVERAI MULTI-AGENT
===============================================================================

                              PATIENT DATA
                                   |
                                   v
                    +---------------------------+
                    |    LIVERAI ORCHESTRATOR   |
                    +-------------+-------------+
                                  |
             +--------------------+--------------------+
             |          |         |         |          |
             v          v         v         v          v
          +------+  +--------+ +--------+ +------+ +-------------+
          | FAT  |  |FIBROSIS| |CIRRHOSIS| |TUMOR| |SEGMENTATION|
          +------+  +--------+ +--------+ +------+ +-------------+
             |          |         |         |          |
             +----------+---------+---------+----------+
                                  |
                                  v
                    +---------------------------+
                    | CLINICAL REASONING AGENT  |
                    +-------------+-------------+
                                  |
                                  v
                       UNIFIED LIVER ASSESSMENT

===============================================================================
"""
    )


# =============================================================================
# SYSTEM STATUS
# =============================================================================

def system_status(orchestrator):

    print("\n")
    print("=" * 80)
    print("SYSTEM STATUS")
    print("=" * 80)

    if orchestrator is None:

        print(
            "✗ Orchestrator is None"
        )

        print("=" * 80)

        return

    # Try common orchestrator structures
    if hasattr(
        orchestrator,
        "agents"
    ):

        agents = orchestrator.agents

        for name, agent in agents.items():

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
            "⚠ Orchestrator does not expose"
            " an 'agents' dictionary."
        )

    print("=" * 80)


# =============================================================================
# COMPLETE SYSTEM CHECK
# =============================================================================

def full_system_check():

    print("\n")
    print("=" * 80)
    print("LIVERAI COMPLETE SYSTEM CHECK")
    print("=" * 80)

    # Paths
    path_results = check_paths()

    # Models
    model_results = show_model_inventory()

    # Summary
    print("\n")
    print("=" * 80)
    print("CHECK SUMMARY")
    print("=" * 80)

    print(
        "Paths checked:",
        len(path_results)
    )

    print(
        "Model files found:",
        len(model_results)
    )

    print("=" * 80)

    return {
        "paths": path_results,
        "models": model_results
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("\n")
    print("=" * 80)
    print("STARTING LIVERAI")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # CHECK PATHS
    # -------------------------------------------------------------------------

    check_paths()

    # -------------------------------------------------------------------------
    # INVENTORY
    # -------------------------------------------------------------------------

    show_model_inventory()

    # -------------------------------------------------------------------------
    # ARCHITECTURE
    # -------------------------------------------------------------------------

    show_architecture()

    # -------------------------------------------------------------------------
    # ORCHESTRATOR
    # -------------------------------------------------------------------------

    orchestrator = create_orchestrator()

    # -------------------------------------------------------------------------
    # STATUS
    # -------------------------------------------------------------------------

    system_status(
        orchestrator
    )

    return orchestrator


# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":

    orchestrator = main()
