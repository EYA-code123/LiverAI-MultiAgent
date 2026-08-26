# ==============================================================
# LiverAI-MultiAgent
# MAIN ENTRY POINT
# ==============================================================

import os
import json
import traceback

# ==============================================================
# IMPORT AGENTS
# ==============================================================

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

# ==============================================================
# IMPORT ORCHESTRATOR
# ==============================================================

from orchestrator.liver_orchestrator import (
    LiverAIOrchestrator
)


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
#
# IMPORTANT:
# Replace these paths with the EXACT paths of your models.
#
# =============================================================

FATTY_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "fatty_liver",
    "model.pkl"
)

FIBROSIS_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "fibrosis",
    "model.keras"
)

CIRRHOSIS_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "cirrhosis",
    "model.pkl"
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
# OPTIONAL INPUT DATA
# ==============================================================
#
# Put your test files here when you want to run the system.
#
# =============================================================

ULTRASOUND_IMAGE = None

MRI_IMAGE = None

LIVER_VOLUME = None

CLINICAL_DATA = None


# ==============================================================
# CHECK MODEL PATHS
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

    for name, path in paths.items():

        if os.path.exists(path):

            print(
                f"✓ {name:<25} {path}"
            )

        else:

            print(
                f"⚠ {name:<25} NOT FOUND"
            )
            print(
                f"  → {path}"
            )

    print("=" * 80)


# ==============================================================
# CREATE AGENTS
# ==============================================================

def create_agents():

    print("\n")
    print("=" * 80)
    print("INITIALIZING LIVERAI AGENTS")
    print("=" * 80)

    # ----------------------------------------------------------
    # 1. FATTY LIVER AGENT
    # ----------------------------------------------------------

    print("\n[1/6] Initializing Fatty Liver Agent...")

    fatty_liver_agent = FattyLiverAgent(
        model_path=FATTY_MODEL_PATH
    )

    print("✓ Fatty Liver Agent ready")

    # ----------------------------------------------------------
    # 2. FIBROSIS AGENT
    # ----------------------------------------------------------

    print("\n[2/6] Initializing Fibrosis Agent...")

    fibrosis_agent = FibrosisAgent(
        model_path=FIBROSIS_MODEL_PATH
    )

    print("✓ Fibrosis Agent ready")

    # ----------------------------------------------------------
    # 3. CIRRHOSIS AGENT
    # ----------------------------------------------------------

    print("\n[3/6] Initializing Cirrhosis Agent...")

    cirrhosis_agent = CirrhosisAgent(
        model_path=CIRRHOSIS_MODEL_PATH
    )

    print("✓ Cirrhosis Agent ready")

    # ----------------------------------------------------------
    # 4. TUMOR CLASSIFICATION AGENT
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # 5. LIVER SEGMENTATION AGENT
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # 6. CLINICAL REASONING AGENT
    # ----------------------------------------------------------

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
    print("ALL AGENTS INITIALIZED")
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

    # ----------------------------------------------------------
    # THIS IS WHERE THE AGENTS ARE CONNECTED
    # ----------------------------------------------------------

    orchestrator = LiverAIOrchestrator(

        fatty_agent=fatty_liver_agent,

        fibrosis_agent=fibrosis_agent,

        cirrhosis_agent=cirrhosis_agent,

        tumor_agent=tumor_classification_agent,

        segmentation_agent=liver_segmentation_agent,

        clinical_reasoning_agent=clinical_reasoning_agent
    )

    print("\n✓ LiverAI Orchestrator created")

    print("\nRegistered architecture:")

    print("""
        ┌─────────────────────────────────────────┐
        │          LIVERAI ORCHESTRATOR            │
        └────────────────────┬────────────────────┘
                             │
          ┌──────────┬───────┼───────┬──────────┐
          │          │       │       │          │
          ▼          ▼       ▼       ▼          ▼
       FATty      FIBROSIS CIRRHOSIS TUMOR  SEGMENTATION
        Agent       Agent    Agent   Agent      Agent
          │          │       │       │          │
          └──────────┴───────┴───────┴──────────┘
                             │
                             ▼
                  CLINICAL REASONING
                         AGENT
                             │
                             ▼
                  UNIFIED ASSESSMENT
    """)

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

    print("\n")
    print("=" * 80)
    print("STARTING LIVERAI ANALYSIS")
    print("=" * 80)

    print("\nAvailable inputs:")

    print(
        "  Clinical data       :",
        "✓" if clinical_data is not None
        else "—"
    )

    print(
        "  Ultrasound image    :",
        "✓" if ultrasound_image is not None
        else "—"
    )

    print(
        "  MRI image           :",
        "✓" if mri_image is not None
        else "—"
    )

    print(
        "  Liver volume / NPY  :",
        "✓" if liver_volume is not None
        else "—"
    )

    # ==========================================================
    # CALL ORCHESTRATOR
    # ==========================================================

    result = orchestrator.predict(

        clinical_data=clinical_data,

        ultrasound_image=ultrasound_image,

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

    check_model_paths()

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
    #
    # At the moment the four inputs are None.
    #
    # Replace them with your real data.
    #
    # ==========================================================

    try:

        result = run_analysis(

            orchestrator,

            clinical_data=CLINICAL_DATA,

            ultrasound_image=ULTRASOUND_IMAGE,

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
