# ==============================================================
# LiverAI-MultiAgent
# MAIN ENTRY POINT
# ==============================================================

import os
import json
import traceback
import joblib


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
# BASE DIRECTORY
# ==============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
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

    "SegResNet3D_Liver.pth"
)


# ==============================================================
# INPUTS
# ==============================================================

CLINICAL_DATA = None

ULTRASOUND_IMAGE = None

MRI_IMAGE = None

LIVER_VOLUME = None


# ==============================================================
# CHECK MODEL PATHS
# ==============================================================

def check_model_paths():

    print("\n")

    print("=" * 80)

    print(
        "CHECKING MODEL PATHS"
    )

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

    all_found = True

    for name, path in paths.items():

        if os.path.exists(path):

            print(
                f"✓ {name:<20} "
                f"{path}"
            )

        else:

            print(
                f"✗ {name:<20} "
                f"NOT FOUND"
            )

            print(
                f"  → {path}"
            )

            all_found = False

    print(
        "=" * 80
    )

    return all_found


# ==============================================================
# LOAD MODELS
# ==============================================================

def create_agents():

    print("\n")

    print("=" * 80)

    print(
        "INITIALIZING LIVERAI AGENTS"
    )

    print("=" * 80)

    # ==========================================================
    # 1. FATTY LIVER
    # ==========================================================

    print(
        "\n[1/6] Fatty Liver Agent"
    )

    fatty_model = joblib.load(
        FATTY_MODEL_PATH
    )

    fatty_agent = FattyLiverAgent(
        model=fatty_model
    )

    print(
        "✓ Fatty Liver Agent ready"
    )

    # ==========================================================
    # 2. FIBROSIS
    # ==========================================================

    print(
        "\n[2/6] Fibrosis Agent"
    )

    fibrosis_model = joblib.load(
        FIBROSIS_MODEL_PATH
    )

    fibrosis_agent = FibrosisAgent(
        model=fibrosis_model
    )

    print(
        "✓ Fibrosis Agent ready"
    )

    # ==========================================================
    # 3. CIRRHOSIS
    # ==========================================================

    print(
        "\n[3/6] Cirrhosis Agent"
    )

    cirrhosis_package = joblib.load(
        CIRRHOSIS_MODEL_PATH
    )

    cirrhosis_agent = CirrhosisAgent(
        model_package=
            cirrhosis_package
    )

    print(
        "✓ Cirrhosis Agent ready"
    )

    # ==========================================================
    # 4. TUMOR
    # ==========================================================

    print(
        "\n[4/6] Tumor Classification Agent"
    )

    tumor_agent = TumorClassificationAgent(
        model_path=
            TUMOR_MODEL_PATH
    )

    print(
        "✓ Tumor Agent ready"
    )

    # ==========================================================
    # 5. SEGMENTATION
    # ==========================================================

    print(
        "\n[5/6] Liver Segmentation Agent"
    )

    segmentation_agent = (
        LiverSegmentationAgent(

            model_path=
                SEGMENTATION_MODEL_PATH
        )
    )

    print(
        "✓ Segmentation Agent ready"
    )

    # ==========================================================
    # 6. CLINICAL REASONING
    # ==========================================================

    print(
        "\n[6/6] Clinical Reasoning Agent"
    )

    clinical_reasoning_agent = (
        ClinicalReasoningAgent()
    )

    print(
        "✓ Clinical Reasoning Agent ready"
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "ALL AGENTS INITIALIZED"
    )

    print(
        "=" * 80
    )

    return (

        fatty_agent,

        fibrosis_agent,

        cirrhosis_agent,

        tumor_agent,

        segmentation_agent,

        clinical_reasoning_agent
    )


# ==============================================================
# CREATE ORCHESTRATOR
# ==============================================================

def create_orchestrator():

    (

        fatty_agent,

        fibrosis_agent,

        cirrhosis_agent,

        tumor_agent,

        segmentation_agent,

        clinical_reasoning_agent

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
            clinical_reasoning_agent
    )

    return orchestrator


# ==============================================================
# RUN ANALYSIS
# ==============================================================

def run_analysis(

    orchestrator,

    clinical_data=None,

    ultrasound_image=None,

    mri_image=None,

    liver_volume=None

):

    return orchestrator.predict(

        clinical_data=
            clinical_data,

        ultrasound_image=
            ultrasound_image,

        mri_image=
            mri_image,

        liver_volume=
            liver_volume
    )


# ==============================================================
# DISPLAY
# ==============================================================

def display_result(result):

    print("\n")

    print("=" * 80)

    print(
        "FINAL LIVERAI ASSESSMENT"
    )

    print("=" * 80)

    print(
        json.dumps(

            result,

            indent=4,

            default=str
        )
    )

    print(
        "=" * 80
    )


# ==============================================================
# SAVE RESULT
# ==============================================================

def save_result(

    result,

    output_path=
        "liverai_result.json"

):

    try:

        with open(

            output_path,

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(

                result,

                file,

                indent=4,

                default=str
            )

        print(
            f"\n✓ Result saved: "
            f"{output_path}"
        )

    except Exception as e:

        print(
            f"\n✗ Save error: {e}"
        )


# ==============================================================
# MAIN
# ==============================================================

def main():

    print("\n")

    print("=" * 80)

    print(
        "LIVERAI-MULTIAGENT"
    )

    print(
        "MULTI-AGENT LIVER ANALYSIS"
    )

    print("=" * 80)

    # ==========================================================
    # CHECK MODELS
    # ==========================================================

    check_model_paths()

    # ==========================================================
    # INITIALIZE
    # ==========================================================

    try:

        orchestrator = (
            create_orchestrator()
        )

    except Exception as e:

        print("\n")

        print("=" * 80)

        print(
            "INITIALIZATION ERROR"
        )

        print("=" * 80)

        print(
            str(e)
        )

        traceback.print_exc()

        return

    # ==========================================================
    # RUN
    # ==========================================================

    try:

        result = run_analysis(

            orchestrator,

            clinical_data=
                CLINICAL_DATA,

            ultrasound_image=
                ULTRASOUND_IMAGE,

            mri_image=
                MRI_IMAGE,

            liver_volume=
                LIVER_VOLUME
        )

        # ======================================================
        # DISPLAY
        # ======================================================

        display_result(
            result
        )

        # ======================================================
        # SAVE
        # ======================================================

        save_result(
            result
        )

    except Exception as e:

        print("\n")

        print("=" * 80)

        print(
            "ANALYSIS ERROR"
        )

        print("=" * 80)

        print(
            str(e)
        )

        traceback.print_exc()


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":

    main()
