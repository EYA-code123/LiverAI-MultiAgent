# ==============================================================
# LiverAI-MultiAgent
# MAIN ENTRY POINT
# ==============================================================

import os
import sys
import json
import pickle
import joblib
import traceback
from datetime import datetime


# ==============================================================
# GOOGLE DRIVE / PROJECT PATH
# ==============================================================

BASE_DIR = "/content/drive/MyDrive/LiverAI"

DATASETS_DIR = os.path.join(
    BASE_DIR,
    "datasets"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ==============================================================
# MODEL PATHS
# ==============================================================

FATTY_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "fatty_liver",
    "model.pkl"
)

FIBROSIS_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "fibrosis",
    "model.pkl"
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


# ==============================================================
# DATASET PATHS
# ==============================================================

FATTY_DATASET_DIR = os.path.join(
    DATASETS_DIR,
    "fatty_liver"
)

FIBROSIS_DATASET_DIR = os.path.join(
    DATASETS_DIR,
    "fibrosis"
)

CIRRHOSIS_DATASET_DIR = os.path.join(
    DATASETS_DIR,
    "cirrhosis"
)

TUMOR_DATASET_DIR = os.path.join(
    DATASETS_DIR,
    "tumor"
)

SEGMENTATION_DATASET_DIR = os.path.join(
    DATASETS_DIR,
    "segmentation"
)


# ==============================================================
# PRINT PATHS
# ==============================================================

def print_configuration():

    print("\n")
    print("=" * 80)
    print("LIVERAI CONFIGURATION")
    print("=" * 80)

    print("\nProject:")
    print(BASE_DIR)

    print("\nModels:")

    print(
        "Fatty Liver      :",
        FATTY_MODEL_PATH
    )

    print(
        "Fibrosis         :",
        FIBROSIS_MODEL_PATH
    )

    print(
        "Cirrhosis        :",
        CIRRHOSIS_MODEL_PATH
    )

    print(
        "Tumor            :",
        TUMOR_MODEL_PATH
    )

    print(
        "Segmentation     :",
        SEGMENTATION_MODEL_PATH
    )

    print("\nDatasets:")

    print(
        "Fatty Liver      :",
        FATTY_DATASET_DIR
    )

    print(
        "Fibrosis         :",
        FIBROSIS_DATASET_DIR
    )

    print(
        "Cirrhosis        :",
        CIRRHOSIS_DATASET_DIR
    )

    print(
        "Tumor            :",
        TUMOR_DATASET_DIR
    )

    print(
        "Segmentation     :",
        SEGMENTATION_DATASET_DIR
    )

    print("=" * 80)


# ==============================================================
# CHECK DIRECTORIES
# ==============================================================

def check_directories():

    print("\n")
    print("=" * 80)
    print("CHECKING GOOGLE DRIVE DIRECTORIES")
    print("=" * 80)

    directories = {

        "LiverAI":
            BASE_DIR,

        "Datasets":
            DATASETS_DIR,

        "Models":
            MODELS_DIR,

        "Fatty dataset":
            FATTY_DATASET_DIR,

        "Fibrosis dataset":
            FIBROSIS_DATASET_DIR,

        "Cirrhosis dataset":
            CIRRHOSIS_DATASET_DIR,

        "Tumor dataset":
            TUMOR_DATASET_DIR,

        "Segmentation dataset":
            SEGMENTATION_DATASET_DIR,

        "Fatty model directory":
            os.path.dirname(FATTY_MODEL_PATH),

        "Fibrosis model directory":
            os.path.dirname(FIBROSIS_MODEL_PATH),

        "Cirrhosis model directory":
            os.path.dirname(CIRRHOSIS_MODEL_PATH),

        "Tumor model directory":
            os.path.dirname(TUMOR_MODEL_PATH),

        "Segmentation model directory":
            os.path.dirname(SEGMENTATION_MODEL_PATH)
    }

    for name, path in directories.items():

        if os.path.exists(path):

            print(
                f"✓ {name:<30} {path}"
            )

        else:

            print(
                f"✗ {name:<30} NOT FOUND"
            )

    print("=" * 80)


# ==============================================================
# CHECK MODEL FILES
# ==============================================================

def check_models():

    print("\n")
    print("=" * 80)
    print("CHECKING TRAINED MODELS")
    print("=" * 80)

    models = {

        "Fatty Liver":
            FATTY_MODEL_PATH,

        "Fibrosis":
            FIBROSIS_MODEL_PATH,

        "Cirrhosis":
            CIRRHOSIS_MODEL_PATH,

        "Tumor Classification":
            TUMOR_MODEL_PATH,

        "Liver Segmentation":
            SEGMENTATION_MODEL_PATH
    }

    available = {}

    for name, path in models.items():

        exists = os.path.isfile(path)

        available[name] = exists

        if exists:

            size_mb = (
                os.path.getsize(path)
                / (1024 * 1024)
            )

            print(
                f"✓ {name:<25} "
                f"{path} "
                f"({size_mb:.2f} MB)"
            )

        else:

            print(
                f"✗ {name:<25} "
                f"{path}"
            )

    print("=" * 80)

    return available


# ==============================================================
# LOAD PICKLE / JOBLIB MODEL
# ==============================================================

def load_pickle_model(path):

    if not os.path.isfile(path):

        raise FileNotFoundError(
            f"\nModel not found:\n{path}"
        )

    print(
        f"\nLoading model:\n{path}"
    )

    try:

        model = joblib.load(path)

        print("✓ Loaded with joblib")

        return model

    except Exception as joblib_error:

        print(
            "joblib failed, trying pickle..."
        )

        try:

            with open(
                path,
                "rb"
            ) as f:

                model = pickle.load(f)

            print(
                "✓ Loaded with pickle"
            )

            return model

        except Exception as pickle_error:

            raise RuntimeError(
                "\nCould not load model.\n"
                f"Path: {path}\n"
                f"joblib error: {joblib_error}\n"
                f"pickle error: {pickle_error}"
            )


# ==============================================================
# LOAD KERAS MODEL
# ==============================================================

def load_keras_model(path):

    if not os.path.isfile(path):

        raise FileNotFoundError(
            f"\nKeras model not found:\n{path}"
        )

    print(
        f"\nLoading Keras model:\n{path}"
    )

    import tensorflow as tf

    model = tf.keras.models.load_model(
        path,
        compile=False
    )

    print("✓ Keras model loaded")

    return model


# ==============================================================
# IMPORT AGENTS
# ==============================================================

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


# ==============================================================
# IMPORT ORCHESTRATOR
# ==============================================================

from orchestrator.liver_orchestrator import (
    LiverAIOrchestrator
)


# ==============================================================
# CREATE AGENTS
# ==============================================================

def create_agents():

    print("\n")
    print("=" * 80)
    print("INITIALIZING LIVERAI AGENTS")
    print("=" * 80)


    # ==========================================================
    # 1. FATTY LIVER AGENT
    # ==========================================================

    print(
        "\n[1/6] Initializing Fatty Liver Agent..."
    )

    fatty_model = load_pickle_model(
        FATTY_MODEL_PATH
    )

    fatty_liver_agent = FattyLiverAgent(
        model=fatty_model
    )

    print(
        "✓ Fatty Liver Agent ready"
    )


    # ==========================================================
    # 2. FIBROSIS AGENT
    # ==========================================================

    print(
        "\n[2/6] Initializing Fibrosis Agent..."
    )

    fibrosis_model = load_pickle_model(
        FIBROSIS_MODEL_PATH
    )

    fibrosis_agent = FibrosisAgent(
        model=fibrosis_model
    )

    print(
        "✓ Fibrosis Agent ready"
    )


    # ==========================================================
    # 3. CIRRHOSIS AGENT
    # ==========================================================

    print(
        "\n[3/6] Initializing Cirrhosis Agent..."
    )

    cirrhosis_package = load_pickle_model(
        CIRRHOSIS_MODEL_PATH
    )

    if not isinstance(
        cirrhosis_package,
        dict
    ):

        raise TypeError(
            "\nCirrhosis model must be "
            "a dictionary/package."
        )


    required_keys = [

        "model",

        "feature_names",

        "numerical_columns",

        "categorical_columns",

        "numerical_imputer",

        "categorical_imputer"
    ]


    missing_keys = [

        key

        for key in required_keys

        if key not in cirrhosis_package
    ]


    if missing_keys:

        raise KeyError(
            "\nCirrhosis model package "
            "is missing:\n"
            f"{missing_keys}"
        )


    cirrhosis_agent = CirrhosisAgent(
        model_package=cirrhosis_package
    )

    print(
        "✓ Cirrhosis Agent ready"
    )


    # ==========================================================
    # 4. TUMOR CLASSIFICATION AGENT
    # ==========================================================

    print(
        "\n[4/6] Initializing "
        "Tumor Classification Agent..."
    )

    tumor_classification_agent = (
        TumorClassificationAgent(
            model_path=TUMOR_MODEL_PATH
        )
    )

    print(
        "✓ Tumor Classification Agent ready"
    )


    # ==========================================================
    # 5. LIVER SEGMENTATION AGENT
    # ==========================================================

    print(
        "\n[5/6] Initializing "
        "Liver Segmentation Agent..."
    )

    liver_segmentation_agent = (
        LiverSegmentationAgent(
            model_path=SEGMENTATION_MODEL_PATH
        )
    )

    print(
        "✓ Liver Segmentation Agent ready"
    )


    # ==========================================================
    # 6. CLINICAL REASONING AGENT
    # ==========================================================

    print(
        "\n[6/6] Initializing "
        "Clinical Reasoning Agent..."
    )

    clinical_reasoning_agent = (
        ClinicalReasoningAgent()
    )

    print(
        "✓ Clinical Reasoning Agent ready"
    )


    print("\n" + "=" * 80)

    print(
        "ALL AGENTS INITIALIZED"
    )

    print("=" * 80)


    return (

        fatty_liver_agent,

        fibrosis_agent,

        cirrhosis_agent,

        tumor_classification_agent,

        liver_segmentation_agent,

        clinical_reasoning_agent
    )


# ==============================================================
# CREATE ORCHESTRATOR
# ==============================================================

def create_orchestrator():

    print("\n")
    print("=" * 80)
    print("CREATING LIVERAI ORCHESTRATOR")
    print("=" * 80)


    (
        fatty_liver_agent,

        fibrosis_agent,

        cirrhosis_agent,

        tumor_classification_agent,

        liver_segmentation_agent,

        clinical_reasoning_agent

    ) = create_agents()


    # ==========================================================
    # CONNECT ALL AGENTS
    # ==========================================================

    orchestrator = LiverAIOrchestrator(

        fatty_agent=fatty_liver_agent,

        fibrosis_agent=fibrosis_agent,

        cirrhosis_agent=cirrhosis_agent,

        tumor_agent=tumor_classification_agent,

        segmentation_agent=liver_segmentation_agent,

        clinical_reasoning_agent=clinical_reasoning_agent
    )


    print(
        "\n✓ LiverAI Orchestrator created"
    )


    print("\n")
    print("=" * 80)
    print("REGISTERED MULTI-AGENT ARCHITECTURE")
    print("=" * 80)

    print("""
                    ┌───────────────────────────┐
                    │    LIVERAI ORCHESTRATOR   │
                    └─────────────┬─────────────┘
                                  │
             ┌────────────────────┼────────────────────┐
             │                    │                    │
             ▼                    ▼                    ▼
      FATYY LIVER             FIBROSIS             CIRRHOSIS
         AGENT                 AGENT                 AGENT
             │                    │                    │
             └────────────────────┼────────────────────┘
                                  │
             ┌────────────────────┼────────────────────┐
             │                    │                    │
             ▼                    ▼                    ▼
           TUMOR             SEGMENTATION        CLINICAL
          AGENT                  AGENT           REASONING
             │                    │                    │
             └────────────────────┼────────────────────┘
                                  │
                                  ▼
                         UNIFIED ASSESSMENT
    """)

    print("=" * 80)

    return orchestrator


# ==============================================================
# DISPLAY ORCHESTRATOR
# ==============================================================

def show_orchestrator(orchestrator):

    print("\n")
    print("=" * 80)
    print("ORCHESTRATOR STATUS")
    print("=" * 80)

    print(
        "Name:",
        orchestrator.name
    )

    print("\nAgents:")

    for name, agent in (
        orchestrator.agents.items()
    ):

        print(
            f"  ✓ {name:<25} "
            f"{agent.__class__.__name__}"
        )

    print("=" * 80)


# ==============================================================
# RUN ONE ANALYSIS
# ==============================================================

def run_analysis(
    orchestrator,
    clinical_data=None,
    ultrasound_image=None,
    tumor_image=None,
    segmentation_volume=None
):

    print("\n")
    print("=" * 80)
    print("STARTING LIVERAI ANALYSIS")
    print("=" * 80)


    # ----------------------------------------------------------
    # The exact input format is delegated to the orchestrator.
    # ----------------------------------------------------------

    try:

        result = orchestrator.predict(

            clinical_data=clinical_data,

            ultrasound_image=ultrasound_image,

            tumor_image=tumor_image,

            segmentation_volume=segmentation_volume
        )

        print("\n")
        print("=" * 80)
        print("ANALYSIS COMPLETED")
        print("=" * 80)

        return result


    except TypeError:

        # ------------------------------------------------------
        # Compatibility fallback:
        # Your current orchestrator may use different argument
        # names. Try the generic input format.
        # ------------------------------------------------------

        try:

            result = orchestrator.predict({

                "clinical_data":
                    clinical_data,

                "ultrasound_image":
                    ultrasound_image,

                "tumor_image":
                    tumor_image,

                "segmentation_volume":
                    segmentation_volume
            })

            print("\n")
            print("=" * 80)
            print("ANALYSIS COMPLETED")
            print("=" * 80)

            return result

        except Exception as e:

            print(
                "\n✗ Analysis failed:"
            )

            print(e)

            traceback.print_exc()

            return None


    except Exception as e:

        print(
            "\n✗ Analysis failed:"
        )

        print(e)

        traceback.print_exc()

        return None


# ==============================================================
# SAVE RESULT
# ==============================================================

def save_result(
    result,
    output_path=None
):

    if result is None:

        print(
            "No result to save."
        )

        return


    if output_path is None:

        output_path = os.path.join(
            BASE_DIR,
            "liverai_result.json"
        )


    try:

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                result,
                f,
                indent=4,
                ensure_ascii=False,
                default=str
            )


        print(
            f"\n✓ Result saved:\n"
            f"{output_path}"
        )


    except Exception as e:

        print(
            f"\n✗ Could not save result: {e}"
        )


# ==============================================================
# MAIN
# ==============================================================

def main():

    print("\n")
    print("=" * 80)
    print("LIVERAI MULTI-AGENT SYSTEM")
    print("=" * 80)


    # ----------------------------------------------------------
    # 1. Verify Google Drive
    # ----------------------------------------------------------

    if not os.path.exists(BASE_DIR):

        raise FileNotFoundError(
            "\nLiverAI directory does not exist:\n"
            f"{BASE_DIR}\n\n"
            "Mount Google Drive first and make sure "
            "the LiverAI directory was created."
        )


    # ----------------------------------------------------------
    # 2. Configuration
    # ----------------------------------------------------------

    print_configuration()


    # ----------------------------------------------------------
    # 3. Directory check
    # ----------------------------------------------------------

    check_directories()


    # ----------------------------------------------------------
    # 4. Model check
    # ----------------------------------------------------------

    available_models = check_models()


    # ----------------------------------------------------------
    # 5. Stop before creating agents if models are missing
    # ----------------------------------------------------------

    missing_models = [

        name

        for name, available

        in available_models.items()

        if not available
    ]


    if missing_models:

        print("\n")
        print("=" * 80)
        print("MISSING TRAINED MODELS")
        print("=" * 80)

        for name in missing_models:

            print(
                f"✗ {name}"
            )

        print("=" * 80)

        print(
            "\nThe orchestrator cannot initialize "
            "all six agents until these model files "
            "are copied to Google Drive."
        )

        return None


    # ----------------------------------------------------------
    # 6. Create orchestrator
    # ----------------------------------------------------------

    orchestrator = create_orchestrator()


    # ----------------------------------------------------------
    # 7. Show status
    # ----------------------------------------------------------

    show_orchestrator(
        orchestrator
    )


    print("\n")
    print("=" * 80)
    print("LIVERAI READY")
    print("=" * 80)

    print(
        "\nAll six agents are connected "
        "to the orchestrator."
    )

    return orchestrator


# ==============================================================
# EXECUTION
# ==============================================================

if __name__ == "__main__":

    main()
