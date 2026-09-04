# =============================================================================
# LiverAI-MultiAgent
# MAIN ENTRY POINT
# =============================================================================

import os
import sys
import traceback

# =============================================================================
# GOOGLE DRIVE
# =============================================================================

try:

    from google.colab import drive

    drive.mount(
        "/content/drive",
        force_remount=False
    )

except Exception:

    print(
        "Google Drive mount skipped."
    )


# =============================================================================
# PROJECT
# =============================================================================

PROJECT_DIR = (
    "/content/LiverAI-MultiAgent"
)

DRIVE_BASE = (
    "/content/drive/MyDrive"
)

if PROJECT_DIR not in sys.path:

    sys.path.insert(
        0,
        PROJECT_DIR
    )

try:

    os.chdir(
        PROJECT_DIR
    )

except Exception:

    print(
        "WARNING: Project directory "
        "could not be opened:"
    )

    print(
        PROJECT_DIR
    )


print("=" * 80)
print("LIVERAI MULTI-AGENT SYSTEM")
print("=" * 80)

print(
    "Project:",
    PROJECT_DIR
)

print(
    "Drive:",
    DRIVE_BASE
)

print("=" * 80)


# =============================================================================
# MODEL PATHS
# =============================================================================

FATTY_MODEL = (
    DRIVE_BASE
    + "/FattyLiver Agent/"
    + "models/fatty_liver.pkl"
)

FIBROSIS_MODEL = (
    DRIVE_BASE
    + "/Fibrosis Agent/"
    + "XGBoost_model/"
    + "xgboost_nafld.pkl"
)

CIRRHOSIS_MODEL = (
    DRIVE_BASE
    + "/.Cirrhosis Agent/"
    + "XGBoost_model/"
    + "XGBoost_Cirrhosis.pkl"
)

TUMOR_MODEL = (
    DRIVE_BASE
    + "/LiverAI/"
    + "models/tumor/"
    + "model.keras"
)

SEGMENTATION_MODEL = (
    DRIVE_BASE
    + "/LiverAI/"
    + "models/segmentation/"
    + "model.pth"
)


# =============================================================================
# DATA PATHS
# =============================================================================

FATTY_ROOT = (
    DRIVE_BASE
    + "/FattyLiver Agent"
)

FATTY_DATA = (
    DRIVE_BASE
    + "/FattyLiver Agent/DATA"
)

FIBROSIS_ROOT = (
    DRIVE_BASE
    + "/Fibrosis Agent"
)

CIRRHOSIS_ROOT = (
    DRIVE_BASE
    + "/.Cirrhosis Agent"
)

CIRRHOSIS_DATA = (
    DRIVE_BASE
    + "/.Cirrhosis Agent/"
    + "DATA/liver_cirrhosis.csv"
)

TUMOR_ROOT = (
    DRIVE_BASE
    + "/Liver CT Image Dataset"
)

SEGMENTATION_ROOT = (
    DRIVE_BASE
    + "/archive (2)/image"
)


# =============================================================================
# PATH CHECK
# =============================================================================

def check_paths():

    print()
    print("=" * 80)
    print("CHECKING LIVERAI PATHS")
    print("=" * 80)

    paths = {

        "project":
            PROJECT_DIR,

        "fatty_root":
            FATTY_ROOT,

        "fatty_model":
            FATTY_MODEL,

        "fibrosis_root":
            FIBROSIS_ROOT,

        "fibrosis_model":
            FIBROSIS_MODEL,

        "cirrhosis_root":
            CIRRHOSIS_ROOT,

        "cirrhosis_model":
            CIRRHOSIS_MODEL,

        "cirrhosis_data":
            CIRRHOSIS_DATA,

        "tumor_root":
            TUMOR_ROOT,

        "tumor_model":
            TUMOR_MODEL,

        "segmentation_root":
            SEGMENTATION_ROOT,

        "segmentation_model":
            SEGMENTATION_MODEL,
    }

    for name, path in paths.items():

        status = (
            "FOUND"
            if os.path.exists(path)
            else "MISSING"
        )

        print(
            f"{name:<25}: "
            f"{status}"
        )

        print(
            " " * 29,
            path
        )

    print("=" * 80)

    return paths


# =============================================================================
# LOAD PICKLE
# =============================================================================

def load_pickle_model(
    path
):

    import pickle

    if not os.path.exists(
        path
    ):

        raise FileNotFoundError(
            path
        )

    with open(
        path,
        "rb"
    ) as f:

        return pickle.load(
            f
        )


# =============================================================================
# CREATE FATTY LIVER AGENT
# =============================================================================

def create_fatty_agent():

    try:

        from agents.fatty_liver_agent import (
            FattyLiverAgent
        )

        package = load_pickle_model(
            FATTY_MODEL
        )

        # Try model package first
        try:

            agent = FattyLiverAgent(
                model_package=package
            )

        except TypeError:

            agent = FattyLiverAgent(
                model=package
            )

        print(
            "✓ Fatty Liver Agent initialized"
        )

        return agent

    except Exception as e:

        print(
            "✗ Fatty Liver Agent failed:",
            e
        )

        traceback.print_exc()

        return None


# =============================================================================
# CREATE FIBROSIS AGENT
# =============================================================================

def create_fibrosis_agent():

    try:

        from agents.fibrosis_agent import (
            FibrosisAgent
        )

        package = load_pickle_model(
            FIBROSIS_MODEL
        )

        try:

            agent = FibrosisAgent(
                model_package=package
            )

        except TypeError:

            agent = FibrosisAgent(
                model=package
            )

        print(
            "✓ Fibrosis Agent initialized"
        )

        return agent

    except Exception as e:

        print(
            "✗ Fibrosis Agent failed:",
            e
        )

        traceback.print_exc()

        return None


# =============================================================================
# CREATE CIRRHOSIS AGENT
# =============================================================================

def create_cirrhosis_agent():

    try:

        from agents.cirrhosis_agent import (
            CirrhosisAgent
        )

        package = load_pickle_model(
            CIRRHOSIS_MODEL
        )

        if isinstance(
            package,
            dict
        ):

            agent = CirrhosisAgent(
                model_package=package
            )

        else:

            agent = CirrhosisAgent(
                model=package
            )

        print(
            "✓ Cirrhosis Agent initialized"
        )

        return agent

    except Exception as e:

        print(
            "✗ Cirrhosis Agent failed:",
            e
        )

        traceback.print_exc()

        return None


# =============================================================================
# CREATE TUMOR AGENT
# =============================================================================

def create_tumor_agent():

    try:

        from agents.tumor_classification_agent import (
            TumorClassificationAgent
        )

        agent = TumorClassificationAgent(
            model_path=TUMOR_MODEL
        )

        print(
            "✓ Tumor Classification Agent initialized"
        )

        return agent

    except Exception as e:

        print(
            "✗ Tumor Agent failed:",
            e
        )

        traceback.print_exc()

        return None


# =============================================================================
# CREATE SEGMENTATION AGENT
# =============================================================================

def create_segmentation_agent():

    try:

        from agents.liver_segmentation_agent import (
            LiverSegmentationAgent
        )

        agent = LiverSegmentationAgent(
            model_path=SEGMENTATION_MODEL
        )

        print(
            "✓ Liver Segmentation Agent initialized"
        )

        return agent

    except Exception as e:

        print(
            "✗ Segmentation Agent failed:",
            e
        )

        traceback.print_exc()

        return None


# =============================================================================
# CREATE CLINICAL REASONING AGENT
# =============================================================================

def create_clinical_agent():

    try:

        from agents.clinical_reasoning_agent import (
            ClinicalReasoningAgent
        )

        agent = ClinicalReasoningAgent()

        print(
            "✓ Clinical Reasoning Agent initialized"
        )

        return agent

    except Exception as e:

        print(
            "✗ Clinical Reasoning Agent failed:",
            e
        )

        traceback.print_exc()

        return None


# =============================================================================
# CREATE ORCHESTRATOR
# =============================================================================

def create_orchestrator():

    print()
    print("=" * 80)
    print("INITIALIZING LIVERAI ORCHESTRATOR")
    print("=" * 80)

    fatty_agent = (
        create_fatty_agent()
    )

    fibrosis_agent = (
        create_fibrosis_agent()
    )

    cirrhosis_agent = (
        create_cirrhosis_agent()
    )

    tumor_agent = (
        create_tumor_agent()
    )

    segmentation_agent = (
        create_segmentation_agent()
    )

    clinical_agent = (
        create_clinical_agent()
    )

    from orchestrator.liver_orchestrator import (
        LiverAIOrchestrator
    )

    orchestrator = LiverAIOrchestrator(

        cirrhosis_agent=
            cirrhosis_agent,

        fatty_liver_agent=
            fatty_agent,

        clinical_agent=
            clinical_agent,

        fibrosis_agent=
            fibrosis_agent,

        tumor_agent=
            tumor_agent,

        segmentation_agent=
            segmentation_agent,
    )

    print()
    print(
        "✓ LiverAI Orchestrator initialized"
    )

    return orchestrator


# =============================================================================
# DISPLAY RESULT
# =============================================================================

def print_final_result(
    result
):

    print()
    print("=" * 80)
    print("LIVERAI FINAL RESULT")
    print("=" * 80)

    print(
        "Patient:",
        result.get(
            "patient_id"
        )
    )

    print(
        "System status:",
        result.get(
            "status"
        )
    )

    print(
        "Agents completed:",
        result.get(
            "agents_completed"
        ),
        "/",
        result.get(
            "total_specialized_agents"
        )
    )

    print()

    agents = result.get(
        "agents",
        {}
    )

    for name, data in agents.items():

        print(
            f"{name:<30} "
            f"| "
            f"{data.get('status'):<15}"
        )

        print(
            " " * 32,
            "Prediction:",
            data.get(
                "prediction"
            )
        )

        print(
            " " * 32,
            "Confidence:",
            data.get(
                "confidence"
            )
        )

    print()

    clinical = result.get(
        "clinical_reasoning",
        {}
    )

    print("=" * 80)
    print("CLINICAL REASONING")
    print("=" * 80)

    print(
        "Status:",
        clinical.get(
            "status"
        )
    )

    print(
        "Prediction:",
        clinical.get(
            "prediction"
        )
    )

    print(
        "Summary:",
        clinical.get(
            "summary"
        )
    )

    print()

    decision = result.get(
        "decision",
        {}
    )

    print("=" * 80)
    print("FINAL DECISION")
    print("=" * 80)

    print(
        "Risk level:",
        decision.get(
            "risk_level"
        )
    )

    print(
        "Risk score:",
        decision.get(
            "risk_score"
        )
    )

    print(
        "Decision confidence:",
        decision.get(
            "decision_confidence"
        )
    )

    print(
        "Additional tests:",
        decision.get(
            "request_additional_tests"
        )
    )

    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    paths = check_paths()

    orchestrator = (
        create_orchestrator()
    )

    print()
    print("=" * 80)
    print("SYSTEM STATUS")
    print("=" * 80)

    status = (
        orchestrator.get_system_status()
    )

    for name, info in status.items():

        print(
            f"{name:<30}: "
            f"{'READY' if info['loaded'] else 'NOT READY'}"
        )

    print()
    print("=" * 80)
    print("SYSTEM READY")
    print("=" * 80)

    print(
        """
Use:

result = orchestrator.run(
    patient_id="patient_001",
    clinical_data=clinical_data,
    fibrosis_input=fibrosis_input,
    image=mri_image,
    volume=liver_volume
)

print_final_result(result)
"""
    )
