# ================================================================
# WRITE CORRECTED main.py
# ================================================================

PROJECT_DIR = "/content/LiverAI-MultiAgent"

main_code = r'''
# =============================================================================
# LiverAI-MultiAgent
# MAIN ENTRY POINT - CORRECTED VERSION
# =============================================================================

import os
import sys
import glob
import pickle
import joblib
import traceback

# =============================================================================
# PROJECT
# =============================================================================

PROJECT_DIR = "/content/LiverAI-MultiAgent"
DRIVE_BASE = "/content/drive/MyDrive"

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


# =============================================================================
# REAL DATA PATHS
# =============================================================================

FATTY_ROOT = "/content/drive/MyDrive/FattyLiver Agent"
FATTY_DATA = "/content/drive/MyDrive/FattyLiver Agent/DATA"

FIBROSIS_ROOT = "/content/drive/MyDrive/Fibrosis Agent"
FIBROSIS_MODEL = (
    "/content/drive/MyDrive/Fibrosis Agent/"
    "XGBoost_model/xgboost_nafld.pkl"
)

CIRRHOSIS_ROOT = "/content/drive/MyDrive/.Cirrhosis Agent"
CIRRHOSIS_DATA = (
    "/content/drive/MyDrive/.Cirrhosis Agent/"
    "DATA/liver_cirrhosis.csv"
)

# Expected location after saving the trained cirrhosis model
CIRRHOSIS_MODEL = (
    "/content/drive/MyDrive/.Cirrhosis Agent/"
    "XGBoost_model/XGBoost_Cirrhosis.pkl"
)

TUMOR_ROOT = (
    "/content/drive/MyDrive/"
    "Liver CT Image Dataset"
)

SEGMENTATION_ROOT = (
    "/content/drive/MyDrive/"
    "archive (2)/image"
)

# Expected model locations
TUMOR_MODEL = (
    "/content/drive/MyDrive/LiverAI/"
    "models/tumor/model.keras"
)

SEGMENTATION_MODEL = (
    "/content/drive/MyDrive/LiverAI/"
    "models/segmentation/model.pth"
)

# Expected fatty liver model locations
FATTY_MODEL = (
    "/content/drive/MyDrive/FattyLiver Agent/"
    "models/fatty_liver.pkl"
)


# =============================================================================
# CHECK PATHS
# =============================================================================

def check_paths():

    print("=" * 70)
    print("CHECKING LIVERAI PATHS")
    print("=" * 70)

    paths = {
        "fatty_root": FATTY_ROOT,
        "fatty_data": FATTY_DATA,
        "fibrosis_root": FIBROSIS_ROOT,
        "fibrosis_model": FIBROSIS_MODEL,
        "cirrhosis_root": CIRRHOSIS_ROOT,
        "cirrhosis_data": CIRRHOSIS_DATA,
        "cirrhosis_model": CIRRHOSIS_MODEL,
        "tumor_root": TUMOR_ROOT,
        "tumor_model": TUMOR_MODEL,
        "segmentation_root": SEGMENTATION_ROOT,
        "segmentation_model": SEGMENTATION_MODEL
    }

    results = {}

    for name, path in paths.items():

        exists = os.path.exists(path)

        results[name] = {
            "path": path,
            "exists": exists
        }

        if exists:
            print(f"✓ {name}")
            print(f"    {path}")
        else:
            print(f"✗ {name}")
            print(f"    {path}")

    print("=" * 70)

    return results


# =============================================================================
# FIND TRAINED MODEL FILES
# =============================================================================

def find_model_files():

    print("=" * 80)
    print("SEARCHING FOR TRAINED MODEL FILES")
    print("=" * 80)

    search_roots = [
        "/content/drive/MyDrive/FattyLiver Agent",
        "/content/drive/MyDrive/Fibrosis Agent",
        "/content/drive/MyDrive/.Cirrhosis Agent",
        "/content/drive/MyDrive/LiverAI",
        "/content/drive/MyDrive/Liver CT Image Dataset",
        "/content/drive/MyDrive/archive (2)"
    ]

    extensions = [
        "*.pkl",
        "*.joblib",
        "*.keras",
        "*.h5",
        "*.pth",
        "*.pt",
        "*.onnx"
    ]

    found = []

    for root in search_roots:

        if not os.path.exists(root):
            continue

        for extension in extensions:

            pattern = os.path.join(
                root,
                "**",
                extension
            )

            files = glob.glob(
                pattern,
                recursive=True
            )

            for file in files:

                if file not in found:
                    found.append(file)

    found.sort()

    for file in found:
        print(file)

    print()
    print("=" * 80)
    print(f"TOTAL MODEL FILES FOUND: {len(found)}")
    print("=" * 80)

    return found


# =============================================================================
# MODEL INVENTORY
# =============================================================================

def show_model_inventory():

    print("=" * 80)
    print("LIVERAI MODEL INVENTORY")
    print("=" * 80)

    models = find_model_files()

    print("=" * 80)

    if len(models) == 0:

        print("NO TRAINED MODEL FILES FOUND")

    else:

        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")

    print("=" * 80)

    return models


# =============================================================================
# LOAD PICKLE / JOBLIB
# =============================================================================

def load_pickle_model(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Model not found:\n{path}"
        )

    print(f"Loading model:")
    print(path)

    try:

        model = joblib.load(path)

    except Exception:

        with open(path, "rb") as f:
            model = pickle.load(f)

    print("✓ Model loaded")
    print("  Type:", type(model))

    return model


# =============================================================================
# LOAD KERAS
# =============================================================================

def load_keras_model(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Keras model not found:\n{path}"
        )

    import tensorflow as tf

    print(f"Loading Keras model:")
    print(path)

    model = tf.keras.models.load_model(
        path,
        compile=False
    )

    print("✓ Keras model loaded")

    return model


# =============================================================================
# FATTIY LIVER
# =============================================================================

def create_fatty_agent():

    if not os.path.exists(FATTY_MODEL):

        print("⚠ Fatty Liver model not found")
        print("  Expected:")
        print(f"  {FATTY_MODEL}")
        return None

    try:

        from agents.fatty_liver_agent import FattyLiverAgent

        model = load_pickle_model(FATTY_MODEL)

        agent = FattyLiverAgent(
            model=model
        )

        print("✓ Fatty Liver Agent initialized")

        return agent

    except Exception as e:

        print("✗ Fatty Liver Agent failed:")
        print(e)
        traceback.print_exc()

        return None


# =============================================================================
# FIBROSIS
# =============================================================================

def create_fibrosis_agent():

    try:

        from agents.fibrosis_agent import FibrosisAgent

        model = load_pickle_model(
            FIBROSIS_MODEL
        )

        print()
        print("Fibrosis model information:")

        if hasattr(model, "feature_names_in_"):
            print(
                "FEATURES:",
                list(model.feature_names_in_)
            )

        if hasattr(model, "n_features_in_"):
            print(
                "N FEATURES:",
                model.n_features_in_
            )

        if hasattr(model, "classes_"):
            print(
                "CLASSES:",
                model.classes_
            )

        agent = FibrosisAgent(
            model=model
        )

        print("✓ Fibrosis Agent initialized")

        return agent

    except Exception as e:

        print("✗ Fibrosis Agent failed:")
        print(e)
        traceback.print_exc()

        return None


# =============================================================================
# CIRRHOSIS
# =============================================================================

def create_cirrhosis_agent():

    if not os.path.exists(CIRRHOSIS_MODEL):

        print("⚠ Cirrhosis model not found")
        print("  Expected:")
        print(f"  {CIRRHOSIS_MODEL}")
        return None

    try:

        from agents.cirrhosis_agent import CirrhosisAgent

        package = load_pickle_model(
            CIRRHOSIS_MODEL
        )

        print(
            "Cirrhosis artifact type:",
            type(package)
        )

        if not isinstance(package, dict):

            raise TypeError(
                "Cirrhosis artifact must be a dictionary."
            )

        required_keys = [
            "model",
            "feature_names",
            "categorical_columns",
            "numerical_columns"
        ]

        missing = [
            key
            for key in required_keys
            if key not in package
        ]

        if missing:

            raise KeyError(
                f"Missing cirrhosis keys: {missing}"
            )

        agent = CirrhosisAgent(
            model_package=package
        )

        print("✓ Cirrhosis Agent initialized")

        return agent

    except Exception as e:

        print("✗ Cirrhosis Agent failed:")
        print(e)
        traceback.print_exc()

        return None


# =============================================================================
# TUMOR
# =============================================================================

def create_tumor_agent():

    if not os.path.exists(TUMOR_MODEL):

        print("⚠ Tumor model not found")
        print("  Expected:")
        print(f"  {TUMOR_MODEL}")
        return None

    try:

        from agents.tumor_classification_agent import (
            TumorClassificationAgent
        )

        agent = TumorClassificationAgent(
            model_path=TUMOR_MODEL
        )

        print("✓ Tumor Classification Agent initialized")

        return agent

    except Exception as e:

        print("✗ Tumor Agent failed:")
        print(e)
        traceback.print_exc()

        return None


# =============================================================================
# SEGMENTATION
# =============================================================================

def create_segmentation_agent():

    if not os.path.exists(SEGMENTATION_MODEL):

        print("⚠ Segmentation model not found")
        print("  Expected:")
        print(f"  {SEGMENTATION_MODEL}")
        return None

    try:

        from agents.liver_segmentation_agent import (
            LiverSegmentationAgent
        )

        agent = LiverSegmentationAgent(
            model_path=SEGMENTATION_MODEL
        )

        print("✓ Liver Segmentation Agent initialized")

        return agent

    except Exception as e:

        print("✗ Segmentation Agent failed:")
        print(e)
        traceback.print_exc()

        return None


# =============================================================================
# CLINICAL REASONING
# =============================================================================

def create_clinical_reasoning_agent():

    try:

        from agents.clinical_reasoning_agent import (
            ClinicalReasoningAgent
        )

        agent = ClinicalReasoningAgent()

        print("✓ Clinical Reasoning Agent initialized")

        return agent

    except Exception as e:

        print("✗ Clinical Reasoning Agent failed:")
        print(e)

        return None


# =============================================================================
# CREATE ALL AGENTS
# =============================================================================

def create_agents():

    print()
    print("=" * 80)
    print("INITIALIZING ALL LIVERAI AGENTS")
    print("=" * 80)

    fatty_agent = create_fatty_agent()
    fibrosis_agent = create_fibrosis_agent()
    cirrhosis_agent = create_cirrhosis_agent()
    tumor_agent = create_tumor_agent()
    segmentation_agent = create_segmentation_agent()
    clinical_agent = create_clinical_reasoning_agent()

    agents = {

        "fatty_liver": fatty_agent,
        "fibrosis": fibrosis_agent,
        "cirrhosis": cirrhosis_agent,
        "tumor": tumor_agent,
        "segmentation": segmentation_agent,
        "clinical_reasoning": clinical_agent

    }

    print()
    print("=" * 80)
    print("INITIALIZATION SUMMARY")
    print("=" * 80)

    initialized = 0

    for name, agent in agents.items():

        if agent is None:

            print(f"✗ {name}")

        else:

            print(f"✓ {name}")
            initialized += 1

    print()
    print(
        f"Initialized: {initialized}/{len(agents)}"
    )

    print("=" * 80)

    return agents


# =============================================================================
# ORCHESTRATOR
# =============================================================================

def create_orchestrator():

    from orchestrator.liver_orchestrator import (
        LiverAIOrchestrator
    )

    agents = create_agents()

    orchestrator = LiverAIOrchestrator(

        fatty_agent=agents["fatty_liver"],

        fibrosis_agent=agents["fibrosis"],

        cirrhosis_agent=agents["cirrhosis"],

        tumor_agent=agents["tumor"],

        segmentation_agent=agents["segmentation"],

        clinical_reasoning_agent=agents["clinical_reasoning"]

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

    print("""
===============================================================
                    LIVERAI MULTI-AGENT
===============================================================

                       PATIENT DATA
                            |
                            v
                 +----------------------+
                 |  LIVERAI ORCHESTRATOR |
                 +----------+-----------+
                            |
       +----------+---------+---------+----------+
       |          |         |         |          |
       v          v         v         v          v
     FATTY     FIBROSIS  CIRRHOSIS  TUMOR   SEGMENTATION
       |          |         |         |          |
       +----------+---------+---------+----------+
                            |
                            v
                 +--------------------------+
                 | CLINICAL REASONING AGENT |
                 +------------+-------------+
                              |
                              v
                   UNIFIED LIVER ASSESSMENT

===============================================================
""")


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
                print(f"✗ {name:<25} NOT INITIALIZED")
            else:
                print(f"✓ {name:<25} READY")

    else:

        print("⚠ Orchestrator has no 'agents' attribute.")

    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():

    check_paths()

    show_model_inventory()

    show_architecture()

    orchestrator = create_orchestrator()

    system_status(orchestrator)

    return orchestrator


# =============================================================================
# END
# =============================================================================
'''

main_path = os.path.join(PROJECT_DIR, "main.py")

with open(main_path, "w", encoding="utf-8") as f:
    f.write(main_code)

print("=" * 80)
print("✓ main.py REPLACED")
print("=" * 80)
print(main_path)
