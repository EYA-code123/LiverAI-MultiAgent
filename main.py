import os
import sys
import pickle
import joblib
import traceback


# ============================================================
# PATHS
# ============================================================

DRIVE_BASE = "/content/drive/MyDrive"

LIVERAI_DIR = os.path.join(
    DRIVE_BASE,
    "LiverAI"
)

MODELS_DIR = os.path.join(
    LIVERAI_DIR,
    "models"
)

PROJECT_DIR = "/content/LiverAI-MultiAgent"

if os.path.exists(PROJECT_DIR):

    if PROJECT_DIR not in sys.path:

        sys.path.insert(
            0,
            PROJECT_DIR
        )

    os.chdir(
        PROJECT_DIR
    )


# ============================================================
# IMPORTS
# ============================================================

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


# ============================================================
# MODEL PATHS
# ============================================================

FATTY_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "fatty_liver",
    "model.pkl"
)

FIBROSIS_MODEL_PATH = (
    "/content/drive/MyDrive/"
    "Fibrosis Agent/"
    "XGBoost_model/"
    "xgboost_nafld.pkl"
)

CIRRHOSIS_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "cirrhosis",
    "XGBoost_Cirrhosis.pkl"
)

TUMOR_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "tumor",
    "model.keras"
)

SEGMENTATION_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "segmentation",
    "model.keras"
)


# ============================================================
# LOAD PICKLE / JOBLIB
# ============================================================

def load_pickle_model(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Model not found:\n{path}"
        )

    try:

        return joblib.load(path)

    except Exception:

        with open(
            path,
            "rb"
        ) as f:

            return pickle.load(f)


# ============================================================
# CREATE FATTY AGENT
# ============================================================

def create_fatty_agent():

    model = load_pickle_model(
        FATTY_MODEL_PATH
    )

    return FattyLiverAgent(
        model=model
    )


# ============================================================
# CREATE FIBROSIS AGENT
# ============================================================

def create_fibrosis_agent():

    model = load_pickle_model(
        FIBROSIS_MODEL_PATH
    )

    return FibrosisAgent(
        model=model
    )


# ============================================================
# CREATE CIRRHOSIS AGENT
# ============================================================

def create_cirrhosis_agent():

    package = load_pickle_model(
        CIRRHOSIS_MODEL_PATH
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

    return CirrhosisAgent(
        model_package=package
    )


# ============================================================
# CREATE TUMOR AGENT
# ============================================================

def create_tumor_agent():

    return TumorClassificationAgent(
        model_path=TUMOR_MODEL_PATH
    )


# ============================================================
# CREATE SEGMENTATION AGENT
# ============================================================

def create_segmentation_agent():

    return LiverSegmentationAgent(
        model_path=SEGMENTATION_MODEL_PATH
    )


# ============================================================
# CREATE CLINICAL REASONING
# ============================================================

def create_clinical_reasoning_agent():

    return ClinicalReasoningAgent()


# ============================================================
# CREATE ALL AGENTS
# ============================================================

def create_agents():

    agents = {}

    # --------------------------------------------------------
    # FAT
    # --------------------------------------------------------

    try:

        agents["fatty_liver"] = (
            create_fatty_agent()
        )

        print(
            "✓ Fatty Liver Agent"
        )

    except Exception as e:

        print(
            f"✗ Fatty Liver Agent: {e}"
        )

    # --------------------------------------------------------
    # FIBROSIS
    # --------------------------------------------------------

    try:

        agents["fibrosis"] = (
            create_fibrosis_agent()
        )

        print(
            "✓ Fibrosis Agent"
        )

    except Exception as e:

        print(
            f"✗ Fibrosis Agent: {e}"
        )

        traceback.print_exc()

    # --------------------------------------------------------
    # CIRRHOSIS
    # --------------------------------------------------------

    try:

        agents["cirrhosis"] = (
            create_cirrhosis_agent()
        )

        print(
            "✓ Cirrhosis Agent"
        )

    except Exception as e:

        print(
            f"✗ Cirrhosis Agent: {e}"
        )

        traceback.print_exc()

    # --------------------------------------------------------
    # TUMOR
    # --------------------------------------------------------

    try:

        agents["tumor_classification"] = (
            create_tumor_agent()
        )

        print(
            "✓ Tumor Classification Agent"
        )

    except Exception as e:

        print(
            f"✗ Tumor Agent: {e}"
        )

        traceback.print_exc()

    # --------------------------------------------------------
    # SEGMENTATION
    # --------------------------------------------------------

    try:

        agents["liver_segmentation"] = (
            create_segmentation_agent()
        )

        print(
            "✓ Liver Segmentation Agent"
        )

    except Exception as e:

        print(
            f"✗ Segmentation Agent: {e}"
        )

        traceback.print_exc()

    # --------------------------------------------------------
    # CLINICAL REASONING
    # --------------------------------------------------------

    try:

        agents["clinical_reasoning"] = (
            create_clinical_reasoning_agent()
        )

        print(
            "✓ Clinical Reasoning Agent"
        )

    except Exception as e:

        print(
            f"✗ Clinical Reasoning Agent: {e}"
        )

    return agents


# ============================================================
# CREATE ORCHESTRATOR
# ============================================================

def create_orchestrator():

    agents = create_agents()

    orchestrator = LiverAIOrchestrator(

        fatty_agent=
            agents.get(
                "fatty_liver"
            ),

        fibrosis_agent=
            agents.get(
                "fibrosis"
            ),

        cirrhosis_agent=
            agents.get(
                "cirrhosis"
            ),

        tumor_agent=
            agents.get(
                "tumor_classification"
            ),

        segmentation_agent=
            agents.get(
                "liver_segmentation"
            ),

        clinical_reasoning_agent=
            agents.get(
                "clinical_reasoning"
            )
    )

    return orchestrator


# ============================================================
# ARCHITECTURE
# ============================================================

def show_architecture():

    print(
        """
============================================================

                    LIVERAI MULTI-AGENT

                         PATIENT DATA
                              |
                              v
                 +------------------------+
                 |  LIVERAI ORCHESTRATOR  |
                 +-----------+------------+
                             |
          +------------------+------------------+
          |         |          |        |       |
          v         v          v        v       v
       FAT      FIBROSIS   CIRRHOSIS  TUMOR  SEGMENTATION
       AGENT      AGENT      AGENT    AGENT      AGENT
          |         |          |        |       |
          v         v          v        v       v
      LightGBM   XGBoost    XGBoost  Efficient  SegResNet
       / RF       / RF       / RF     /Mobile   / U-Net
          |         |          |        |       |
          +---------+----------+--------+-------+
                             |
                             v
              +-----------------------------+
              | CLINICAL REASONING AGENT    |
              +--------------+--------------+
                             |
                             v
                  +----------------------+
                  | UNIFIED ASSESSMENT   |
                  +----------------------+

============================================================
"""
    )


# ============================================================
# MAIN
# ============================================================

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

    show_architecture()

    orchestrator = create_orchestrator()

    print(
        "\n"
        + "=" * 80
    )

    print(
        "SYSTEM READY"
    )

    print(
        "=" * 80
    )

    print(
        orchestrator.get_status()
    )

    return orchestrator


if __name__ == "__main__":

    orchestrator = main()
