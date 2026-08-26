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

from agents.fatty_liver_agent import FattyLiverAgent
from agents.fibrosis_agent import FibrosisAgent
from agents.cirrhosis_agent import CirrhosisAgent
from agents.tumor_classification_agent import TumorClassificationAgent
from agents.liver_segmentation_agent import LiverSegmentationAgent
from agents.clinical_reasoning_agent import ClinicalReasoningAgent

# ==============================================================
# IMPORT ORCHESTRATOR
# ==============================================================

from orchestrator.liver_orchestrator import LiverAIOrchestrator


# ==============================================================
# PATHS
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

# --------------------------------------------------------------
# FATty LIVER
# --------------------------------------------------------------

FATTY_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "fatty_liver",
    "model.pkl"
)


# --------------------------------------------------------------
# FIBROSIS
# --------------------------------------------------------------
#
# IMPORTANT:
# This is the real model that you already tested successfully.
# --------------------------------------------------------------

FIBROSIS_MODEL_PATH = (
    "/content/drive/MyDrive/"
    "Fibrosis Agent/"
    "XGBoost_model/"
    "xgboost_nafld.pkl"
)


# --------------------------------------------------------------
# CIRRHOSIS
# --------------------------------------------------------------

CIRRHOSIS_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "cirrhosis",
    "model.pkl"
)


# --------------------------------------------------------------
# TUMOR
# --------------------------------------------------------------

TUMOR_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "tumor",
    "model.keras"
)


# --------------------------------------------------------------
# SEGMENTATION
# --------------------------------------------------------------

SEGMENTATION_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "segmentation",
    "model.keras"
)


# ==============================================================
# INPUTS
# ==============================================================

CLINICAL_DATA = None

MRI_IMAGE = None

LIVER_VOLUME = None


# ==============================================================
# CHECK PATHS
# ==============================================================

def check_model_paths():

    print("\n")
    print("=" * 80)
    print("CHECKING MODEL PATHS")
    print("=" * 80)

    paths = {

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

    all_found = True

    for name, path in paths.items():

        if os.path.exists(path):

            print(
                f"✓ {name:<25} {path}"
            )

        else:

            print(
                f"✗ {name:<25} NOT FOUND"
            )

            print(
                f"  → {path}"
            )

            all_found = False

    print("=" * 80)

    return all_found


# ==============================================================
# LOAD MODEL
# ==============================================================

def load_joblib_model(path, name):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"{name} model not found:\n{path}"
        )

    print(
        f"Loading {name}..."
    )

    model = joblib.load(path)

    print(
        f"✓ {name} model loaded: "
        f"{type(model).__name__}"
    )

    return model


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

    fatty_model = load_joblib_model(
        FATTY_MODEL_PATH,
        "Fatty Liver"
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

    fibrosis_model = load_joblib_model(
        FIBROSIS_MODEL_PATH,
        "Fibrosis"
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

    cirrhosis_package = load_joblib_model(
        CIRRHOSIS_MODEL_PATH,
        "Cirrhosis"
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

    if not os.path.exists(
        TUMOR_MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Tumor model not found:\n"
            f"{TUMOR_MODEL_PATH}"
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

    if not os.path.exists(
        SEGMENTATION_MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Segmentation model not found:\n"
            f"{SEGMENTATION_MODEL_PATH}"
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


    print("\n")
    print("=" * 80)
    print("ALL 6 AGENTS INITIALIZED SUCCESSFULLY")
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
    # CONNECT ALL 6 AGENTS
    # ==========================================================

    orchestrator = LiverAIOrchestrator(

        fatty_agent=fatty_liver_agent,

        fibrosis_agent=fibrosis_agent,

        cirrhosis_agent=cirrhosis_agent,

        tumor_agent=tumor_classification_agent,

        segmentation_agent=liver_segmentation_agent,

        clinical_reasoning_agent=clinical_reasoning_agent
    )


    print("\n")
    print("✓ LiverAI Orchestrator created")
    print("✓ 6 agents connected")
    print("=" * 80)


    return orchestrator


# ==============================================================
# RUN ANALYSIS
# ==============================================================

def run_analysis(
    orchestrator,
    clinical_data=None,
    mri_image=None,
    liver_volume=None
):

    print("\n")
    print("=" * 80)
    print("STARTING LIVERAI ANALYSIS")
    print("=" * 80)


    print("\nAvailable inputs:")

    print(
        "  Clinical data      :",
        "✓" if clinical_data is not None
        else "—"
    )

    print(
        "  MRI image          :",
        "✓" if mri_image is not None
        else "—"
    )

    print(
        "  Liver volume / NPY :",
        "✓" if liver_volume is not None
        else "—"
    )


    # ==========================================================
    # CALL ORCHESTRATOR
    # ==========================================================

    result = orchestrator.predict(

        patient_data=clinical_data,

        mri_image=mri_image,

        liver_volume=liver_volume
    )


    return result


# ==============================================================
# DISPLAY RESULT
# ==============================================================

def display_result(result):

    print("\n")
    print("=" * 80)
    print("FINAL LIVERAI ASSESSMENT")
    print("=" * 80)

    print(
        json.dumps(
            result,
            indent=4,
            default=str
        )
    )

    print("=" * 80)


# ==============================================================
# SAVE RESULT
# ==============================================================

def save_result(
    result,
    output_path="liverai_result.json"
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
            f"\n✓ Result saved to: "
            f"{output_path}"
        )

    except Exception as e:

        print(
            f"\n⚠ Could not save result: {e}"
        )


# ==============================================================
# MAIN
# ==============================================================

def main():

    print("\n")
    print("=" * 80)

    print("""
    ██╗     ██╗██╗   ██╗███████╗██████╗  █████╗ ██╗
    ██║     ██║██║   ██║██╔════╝██╔══██╗██╔══██╗██║
    ██║     ██║██║   ██║█████╗  ██████╔╝███████║██║
    ██║     ██║╚██╗ ██╔╝██╔══╝  ██╔══██╗██╔══██║██║
    ███████╗██║ ╚████╔╝ ███████╗██║  ██║██║  ██║██║
    ╚══════╝╚═╝  ╚═══╝  ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝

             MULTI-AGENT LIVER ANALYSIS
    """)

    print("=" * 80)


    # ==========================================================
    # 1. CHECK PATHS
    # ==========================================================

    paths_ok = check_model_paths()

    if not paths_ok:

        print(
            "\n⚠ Some model paths are missing."
        )

        print(
            "Do not start the complete system "
            "until the missing models are available."
        )

        return


    # ==========================================================
    # 2. CREATE ORCHESTRATOR
    # ==========================================================

    try:

        orchestrator = create_orchestrator()

    except Exception as e:

        print("\n")
        print("=" * 80)
        print("ERROR DURING INITIALIZATION")
        print("=" * 80)

        print(str(e))

        traceback.print_exc()

        return


    # ==========================================================
    # 3. RUN ANALYSIS
    # ==========================================================

    try:

        result = run_analysis(

            orchestrator,

            clinical_data=CLINICAL_DATA,

            mri_image=MRI_IMAGE,

            liver_volume=LIVER_VOLUME
        )


        # ======================================================
        # 4. DISPLAY
        # ======================================================

        display_result(result)


        # ======================================================
        # 5. SAVE
        # ======================================================

        save_result(
            result,
            "liverai_result.json"
        )


    except Exception as e:

        print("\n")
        print("=" * 80)
        print("ERROR DURING ANALYSIS")
        print("=" * 80)

        print(str(e))

        traceback.print_exc()


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":

    main()
