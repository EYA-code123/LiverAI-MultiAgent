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
# GOOGLE DRIVE
# =============================================================================

DRIVE_BASE = "/content/drive/MyDrive"

PROJECT_DIR = "/content/LiverAI-MultiAgent"

if PROJECT_DIR not in sys.path:

    sys.path.insert(
        0,
        PROJECT_DIR
    )

os.chdir(
    PROJECT_DIR
)

# =============================================================================
# IMPORT AGENTS
# =============================================================================

from agents.fatty_liver_agent import (
    FattyLiverAgent
)

from agents.fibrosis_agent import (
    FibrosisAgent
)

from agents.cirrhosis_agent import (
    CirrhosisAgent
)

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
# REAL GOOGLE DRIVE MODEL PATHS
# =============================================================================

FATTY_MODEL_PATH = (
    "/content/drive/MyDrive/"
    "FattyLiver Agent/models/"
    "fatty_liver_lightgbm.pkl"
)

FIBROSIS_MODEL_PATH = (
    "/content/drive/MyDrive/"
    "Fibrosis Agent/"
    "XGBoost_model/"
    "xgboost_nafld.pkl"
)

CIRRHOSIS_MODEL_PATH = (
    "/content/drive/MyDrive/"
    ".Cirrhosis Agent/models/"
    "cirrhosis_xgboost.pkl"
)

# These will be updated after your Tumor and Segmentation
# model files are saved.
TUMOR_MODEL_PATH = (
    "/content/drive/MyDrive/"
    "LiverAI_Models/tumor/"
    "model.pth"
)

SEGMENTATION_MODEL_PATH = (
    "/content/drive/MyDrive/"
    "LiverAI_Models/segmentation/"
    "segresnet.pth"
)

# =============================================================================
# MODEL SEARCH
# =============================================================================

def find_existing_model(paths):

    for path in paths:

        if os.path.exists(path):

            return path

    return None


def show_model_paths():

    print("=" * 80)
    print("LIVERAI MODEL STATUS")
    print("=" * 80)

    paths = {

        "Fatty Liver":
            FATTY_MODEL_PATH,

        "Fibrosis":
            FIBROSIS_MODEL_PATH,

        "Cirrhosis":
            CIRRHOSIS_MODEL_PATH,

        "Tumor":
            TUMOR_MODEL_PATH,

        "Segmentation":
            SEGMENTATION_MODEL_PATH
    }

    for name, path in paths.items():

        if os.path.exists(path):

            print(
                f"✓ {name:<20}: "
                f"{path}"
            )

        else:

            print(
                f"✗ {name:<20}: "
                "NOT FOUND"
            )

    print("=" * 80)

# =============================================================================
# LOAD JOBLIB / PICKLE
# =============================================================================

def load_pickle_model(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Model not found:\n{path}"
        )

    try:

        model = joblib.load(
            path
        )

    except Exception:

        with open(
            path,
            "rb"
        ) as f:

            model = pickle.load(
                f
            )

    print(
        "✓ Loaded:",
        path
    )

    print(
        "Type:",
        type(model)
    )

    return model

# =============================================================================
# FATTY LIVER
# =============================================================================

def create_fatty_agent():

    package = load_pickle_model(
        FATTY_MODEL_PATH
    )

    if not isinstance(
        package,
        dict
    ):

        package = {
            "model": package,

            "feature_names": [
                "mcv",
                "alkphos",
                "sgpt",
                "sgot",
                "gammagt",
                "drinks"
            ],

            "model_name":
                "LightGBM"
        }

    agent = FattyLiverAgent(
        model_package=package
    )

    print(
        "✓ Fatty Liver Agent initialized"
    )

    return agent

# =============================================================================
# FIBROSIS
# =============================================================================

def create_fibrosis_agent():

    model = load_pickle_model(
        FIBROSIS_MODEL_PATH
    )

    agent = FibrosisAgent(
        model=model
    )

    print(
        "✓ Fibrosis Agent initialized"
    )

    return agent

# =============================================================================
# CIRRHOSIS
# =============================================================================

def create_cirrhosis_agent():

    package = load_pickle_model(
        CIRRHOSIS_MODEL_PATH
    )

    if not isinstance(
        package,
        dict
    ):

        raise TypeError(
            "Cirrhosis model must be "
            "a preprocessing + model package."
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
            "Missing Cirrhosis package keys: "
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
# TUMOR
# =============================================================================

def create_tumor_agent():

    if not os.path.exists(
        TUMOR_MODEL_PATH
    ):

        print(
            "⚠ Tumor model not found."
        )

        return None

    try:

        agent = TumorClassificationAgent(
            model_path=TUMOR_MODEL_PATH
        )

        print(
            "✓ Tumor Agent initialized"
        )

        return agent

    except Exception as e:

        print(
            "✗ Tumor Agent failed:",
            e
        )

        return None

# =============================================================================
# SEGMENTATION
# =============================================================================

def create_segmentation_agent():

    if not os.path.exists(
        SEGMENTATION_MODEL_PATH
    ):

        print(
            "⚠ Segmentation model not found."
        )

        return None

    try:

        agent = LiverSegmentationAgent(
            model_path=SEGMENTATION_MODEL_PATH
        )

        print(
            "✓ Segmentation Agent initialized"
        )

        return agent

    except Exception as e:

        print(
            "✗ Segmentation Agent failed:",
            e
        )

        return None

# =============================================================================
# CLINICAL REASONING
# =============================================================================

def create_clinical_reasoning_agent():

    agent = ClinicalReasoningAgent()

    print(
        "✓ Clinical Reasoning Agent initialized"
    )

    return agent

# =============================================================================
# CREATE ALL AGENTS
# =============================================================================

def create_agents():

    fatty_agent = None
    fibrosis_agent = None
    cirrhosis_agent = None
    tumor_agent = None
    segmentation_agent = None

    # ------------------------------------------------------
    # FATTY
    # ------------------------------------------------------

    try:

        fatty_agent = (
            create_fatty_agent()
        )

    except Exception as e:

        print(
            "✗ Fatty Liver Agent:",
            e
        )

        traceback.print_exc()

    # ------------------------------------------------------
    # FIBROSIS
    # ------------------------------------------------------

    try:

        fibrosis_agent = (
            create_fibrosis_agent()
        )

    except Exception as e:

        print(
            "✗ Fibrosis Agent:",
            e
        )

        traceback.print_exc()

    # ------------------------------------------------------
    # CIRRHOSIS
    # ------------------------------------------------------

    try:

        cirrhosis_agent = (
            create_cirrhosis_agent()
        )

    except Exception as e:

        print(
            "✗ Cirrhosis Agent:",
            e
        )

        traceback.print_exc()

    # ------------------------------------------------------
    # TUMOR
    # ------------------------------------------------------

    tumor_agent = (
        create_tumor_agent()
    )

    # ------------------------------------------------------
    # SEGMENTATION
    # ------------------------------------------------------

    segmentation_agent = (
        create_segmentation_agent()
    )

    # ------------------------------------------------------
    # CLINICAL
    # ------------------------------------------------------

    clinical_agent = (
        create_clinical_reasoning_agent()
    )

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

    (
        fatty_agent,
        fibrosis_agent,
        cirrhosis_agent,
        tumor_agent,
        segmentation_agent,
        clinical_agent

    ) = create_agents()

    orchestrator = LiverAIOrchestrator(

        fatty_agent=
            fatty_agent,

        fibrosis_agent=
            fibrosis_agent,

        cirrhosis_agent=
            cirrhosis_agent,

        tumor_agent=
            tumor_agent,

        segmentation_agent=
            segmentation_agent,

        clinical_reasoning_agent=
            clinical_agent
    )

    return orchestrator

# =============================================================================
# ARCHITECTURE
# =============================================================================

def show_architecture():

    print(
        """
============================================================

                    LiverAI Orchestrator

============================================================
                         |
       +-----------------+------------------+
       |        |        |        |         |
       v        v        v        v         v

     Fatty   Fibrosis Cirrhosis  Tumor  Segmentation
     Liver     Agent    Agent    Agent      Agent

   LightGBM  XGBoost  XGBoost  EfficientNet SegResNet

       |        |        |        |         |
       +--------+--------+--------+---------+
                         |
                         v
              Clinical Reasoning Agent
                         |
                         v
                 Unified Assessment

============================================================
        """
    )

# =============================================================================
# STATUS
# =============================================================================

def system_status(
    orchestrator
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "LIVERAI SYSTEM STATUS"
    )

    print(
        "=" * 80
    )

    status = (
        orchestrator.get_status()
    )

    for name, info in status.items():

        symbol = (
            "✓"
            if info["available"]
            else "✗"
        )

        print(
            f"{symbol} "
            f"{name:<25} "
            f"{info['class']}"
        )

    print(
        "=" * 80
    )

# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "\n"
        + "=" * 80
    )

    print(
        "STARTING LIVERAI"
    )

    print(
        "=" * 80
    )

    show_model_paths()

    show_architecture()

    orchestrator = (
        create_orchestrator()
    )

    system_status(
        orchestrator
    )

    return orchestrator


if __name__ == "__main__":

    orchestrator = main()
