# =============================================================================
# LiverAI-MultiAgent
# MAIN ENTRY POINT
# =============================================================================

import os
import sys
import json
import pickle
import joblib
import traceback
from datetime import datetime

# =============================================================================
# GOOGLE DRIVE
# =============================================================================

DRIVE_BASE = "/content/drive/MyDrive"

LIVERAI_DIR = os.path.join(
    DRIVE_BASE,
    "LiverAI"
)

MODELS_DIR = os.path.join(
    LIVERAI_DIR,
    "models"
)

DATASETS_DIR = os.path.join(
    LIVERAI_DIR,
    "datasets"
)

# =============================================================================
# PROJECT DIRECTORY
# =============================================================================

PROJECT_DIR = "/content/LiverAI-MultiAgent"

if os.path.exists(PROJECT_DIR):
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)

os.chdir(PROJECT_DIR)

print("=" * 80)
print("LIVERAI MULTI-AGENT SYSTEM")
print("=" * 80)

print("PROJECT:")
print(PROJECT_DIR)

print("GOOGLE DRIVE:")
print(LIVERAI_DIR)

print("=" * 80)


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


# =============================================================================
# IMPORT ORCHESTRATOR
# =============================================================================

from orchestrator.liver_orchestrator import (
    LiverAIOrchestrator
)


# =============================================================================
# REAL MODEL PATHS
# =============================================================================

# -------------------------------------------------------------------------
# FATTY LIVER
# -------------------------------------------------------------------------

FATTY_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "fatty_liver",
    "model.pkl"
)


# -------------------------------------------------------------------------
# FIBROSIS
#
# CURRENT REAL LOCATION FOUND IN YOUR DRIVE
# -------------------------------------------------------------------------

FIBROSIS_MODEL_PATH = (
    "/content/drive/MyDrive/"
    "Fibrosis Agent/"
    "XGBoost_model/"
    "xgboost_nafld.pkl"
)


# -------------------------------------------------------------------------
# CIRRHOSIS
# -------------------------------------------------------------------------

CIRRHOSIS_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "cirrhosis",
    "XGBoost_Cirrhosis.pkl"
)


# -------------------------------------------------------------------------
# TUMOR
# -------------------------------------------------------------------------

TUMOR_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "tumor",
    "model.keras"
)


# -------------------------------------------------------------------------
# SEGMENTATION
# -------------------------------------------------------------------------

SEGMENTATION_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "segmentation",
    "model.keras"
)


# =============================================================================
# DISPLAY MODEL PATHS
# =============================================================================

def show_model_paths():

    print("\n")
    print("=" * 80)
    print("MODEL PATHS")
    print("=" * 80)

    paths = {
        "Fatty Liver": FATTY_MODEL_PATH,
        "Fibrosis": FIBROSIS_MODEL_PATH,
        "Cirrhosis": CIRRHOSIS_MODEL_PATH,
        "Tumor": TUMOR_MODEL_PATH,
        "Segmentation": SEGMENTATION_MODEL_PATH
    }

    for name, path in paths.items():

        if os.path.exists(path):

            print(
                f"✓ {name:<20}: {path}"
            )

        else:

            print(
                f"✗ {name:<20}: NOT FOUND"
            )

    print("=" * 80)


# =============================================================================
# LOAD PICKLE
# =============================================================================

def load_pickle_model(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nModel not found:\n{path}"
        )

    print(
        f"\nLoading pickle model:\n{path}"
    )

    try:

        model = joblib.load(path)

    except Exception:

        with open(
            path,
            "rb"
        ) as f:

            model = pickle.load(f)

    print(
        "✓ Pickle model loaded"
    )

    print(
        "Model type:",
        type(model)
    )

    return model


# =============================================================================
# LOAD KERAS
# =============================================================================

def load_keras_model(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nKeras model not found:\n{path}"
        )

    import tensorflow as tf

    print(
        f"\nLoading Keras model:\n{path}"
    )

    model = tf.keras.models.load_model(
        path,
        compile=False
    )

    print(
        "✓ Keras model loaded"
    )

    return model


# =============================================================================
# LOAD FATTY LIVER
# =============================================================================

def create_fatty_agent():

    print("\n")
    print("-" * 80)
    print("FATTY LIVER AGENT")
    print("-" * 80)

    model = load_pickle_model(
        FATTY_MODEL_PATH
    )

    agent = FattyLiverAgent(
        model=model
    )

    print(
        "✓ Fatty Liver Agent initialized"
    )

    return agent


# =============================================================================
# LOAD FIBROSIS
# =============================================================================

def create_fibrosis_agent():

    print("\n")
    print("-" * 80)
    print("FIBROSIS AGENT")
    print("-" * 80)

    model = load_pickle_model(
        FIBROSIS_MODEL_PATH
    )

    print("\nFibrosis model information:")

    if hasattr(
        model,
        "feature_names_in_"
    ):

        print(
            "FEATURES:",
            list(model.feature_names_in_)
        )

    if hasattr(
        model,
        "n_features_in_"
    ):

        print(
            "N FEATURES:",
            model.n_features_in_
        )

    if hasattr(
        model,
        "classes_"
    ):

        print(
            "CLASSES:",
            model.classes_
        )

    print(
        "MODEL TYPE:",
        type(model)
    )

    agent = FibrosisAgent(
        model=model
    )

    print(
        "✓ Fibrosis Agent initialized"
    )

    return agent


# =============================================================================
# LOAD CIRRHOSIS
# =============================================================================

def create_cirrhosis_agent():

    print("\n")
    print("-" * 80)
    print("CIRRHOSIS AGENT")
    print("-" * 80)

    package = load_pickle_model(
        CIRRHOSIS_MODEL_PATH
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
            "Cirrhosis model must be a dictionary."
        )

    required_keys = [

        "model",
        "feature_names",
        "numerical_columns",
        "categorical_columns",
        "numerical_imputer",
        "categorical_imputer"
    ]

    missing = [

        key
        for key in required_keys
        if key not in package
    ]

    if missing:

        raise KeyError(
            "Missing cirrhosis package keys: "
            + str(missing)
        )

    agent = CirrhosisAgent(
        model_package=package
    )

    print(
        "✓ Cirrhosis Agent initialized"
    )

    return agent


# =============================================================================
# LOAD TUMOR
# =============================================================================

def create_tumor_agent():

    print("\n")
    print("-" * 80)
    print("TUMOR CLASSIFICATION AGENT")
    print("-" * 80)

    model = load_keras_model(
        TUMOR_MODEL_PATH
    )

    agent = TumorClassificationAgent(
        model_path=TUMOR_MODEL_PATH
    )

    print(
        "✓ Tumor Classification Agent initialized"
    )

    return agent


# =============================================================================
# LOAD SEGMENTATION
# =============================================================================

def create_segmentation_agent():

    print("\n")
    print("-" * 80)
    print("LIVER SEGMENTATION AGENT")
    print("-" * 80)

    model = load_keras_model(
        SEGMENTATION_MODEL_PATH
    )

    agent = LiverSegmentationAgent(
        model_path=SEGMENTATION_MODEL_PATH
    )

    print(
        "✓ Liver Segmentation Agent initialized"
    )

    return agent


# =============================================================================
# CLINICAL REASONING
# =============================================================================

def create_clinical_reasoning_agent():

    print("\n")
    print("-" * 80)
    print("CLINICAL REASONING AGENT")
    print("-" * 80)

    agent = ClinicalReasoningAgent()

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

    fatty_agent = None
    fibrosis_agent = None
    cirrhosis_agent = None
    tumor_agent = None
    segmentation_agent = None
    clinical_agent = None

    # -------------------------------------------------------------------------
    # FAT
    # -------------------------------------------------------------------------

    try:

        fatty_agent = create_fatty_agent()

    except Exception as e:

        print(
            f"✗ Fatty Liver Agent failed: {e}"
        )

    # -------------------------------------------------------------------------
    # FIBROSIS
    # -------------------------------------------------------------------------

    try:

        fibrosis_agent = create_fibrosis_agent()

    except Exception as e:

        print(
            f"✗ Fibrosis Agent failed: {e}"
        )

        traceback.print_exc()

    # -------------------------------------------------------------------------
    # CIRRHOSIS
    # -------------------------------------------------------------------------

    try:

        cirrhosis_agent = create_cirrhosis_agent()

    except Exception as e:

        print(
            f"✗ Cirrhosis Agent failed: {e}"
        )

    # -------------------------------------------------------------------------
    # TUMOR
    # -------------------------------------------------------------------------

    try:

        tumor_agent = create_tumor_agent()

    except Exception as e:

        print(
            f"✗ Tumor Agent failed: {e}"
        )

    # -------------------------------------------------------------------------
    # SEGMENTATION
    # -------------------------------------------------------------------------

    try:

        segmentation_agent = (
            create_segmentation_agent()
        )

    except Exception as e:

        print(
            f"✗ Segmentation Agent failed: {e}"
        )

    # -------------------------------------------------------------------------
    # CLINICAL REASONING
    # -------------------------------------------------------------------------

    try:

        clinical_agent = (
            create_clinical_reasoning_agent()
        )

    except Exception as e:

        print(
            f"✗ Clinical Reasoning Agent failed: {e}"
        )

    # -------------------------------------------------------------------------
    # RETURN
    # -------------------------------------------------------------------------

    return (

        fatty_agent,
        fibrosis_agent,
        cirrhosis_agent,
        tumor_agent,
        segmentation_agent,
        clinical_agent
    )


# =============================================================================
# CREATE ORCHESTRATOR
# =============================================================================

def create_orchestrator():

    print("\n")
    print("=" * 80)
    print("CREATING LIVERAI ORCHESTRATOR")
    print("=" * 80)

    (

        fatty_agent,
        fibrosis_agent,
        cirrhosis_agent,
        tumor_agent,
        segmentation_agent,
        clinical_agent

    ) = create_agents()

    # =========================================================================
    # ORCHESTRATOR
    # =========================================================================

    orchestrator = LiverAIOrchestrator(

        fatty_agent=fatty_agent,

        fibrosis_agent=fibrosis_agent,

        cirrhosis_agent=cirrhosis_agent,

        tumor_agent=tumor_agent,

        segmentation_agent=segmentation_agent,

        clinical_reasoning_agent=clinical_agent
    )

    print("\n")
    print("=" * 80)
    print("✓ LIVERAI ORCHESTRATOR CREATED")
    print("=" * 80)

    return orchestrator


# =============================================================================
# ARCHITECTURE DISPLAY
# =============================================================================

def show_architecture():

    print(
        """
        
        ================================================================
                         LIVERAI MULTI-AGENT
        ================================================================

                              PATIENT DATA
                                   |
                                   v
                        +----------------------+
                        |  LIVERAI ORCHESTRATOR |
                        +----------+-----------+
                                   |
              +--------------------+--------------------+
              |         |          |          |          |
              v         v          v          v          v
           FAT/NAFLD FIBROSIS  CIRRHOSIS   TUMOR   SEGMENTATION
              |         |          |          |          |
              |         |          |          |          |
              +---------+----------+----------+----------+
                                   |
                                   v
                      +--------------------------+
                      | CLINICAL REASONING AGENT |
                      +------------+-------------+
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

    print("\n")
    print("=" * 80)
    print("SYSTEM STATUS")
    print("=" * 80)

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

    print("\n")
    print("=" * 80)
    print("STARTING LIVERAI")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # SHOW MODELS
    # -------------------------------------------------------------------------

    show_model_paths()

    # -------------------------------------------------------------------------
    # SHOW ARCHITECTURE
    # -------------------------------------------------------------------------

    show_architecture()

    # -------------------------------------------------------------------------
    # CREATE ORCHESTRATOR
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
