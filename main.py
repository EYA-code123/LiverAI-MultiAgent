# =============================================================================
# LiverAI-MultiAgent
# MAIN ENTRY POINT
# =============================================================================

import os
import sys
import glob
import pickle
import traceback


# =============================================================================
# PROJECT CONFIGURATION
# =============================================================================

PROJECT_DIR = "/content/LiverAI-MultiAgent"

DRIVE_BASE = "/content/drive/MyDrive"


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
        f"WARNING: "
        f"Could not change directory "
        f"to {PROJECT_DIR}"
    )


print("=" * 80)
print("LIVERAI MULTI-AGENT SYSTEM")
print("=" * 80)

print(
    "PROJECT:",
    PROJECT_DIR
)

print(
    "GOOGLE DRIVE:",
    DRIVE_BASE
)

print("=" * 80)


# =============================================================================
# MODEL PATHS
# =============================================================================

FATTY_MODEL = (
    "/content/drive/MyDrive/"
    "FattyLiver Agent/"
    "models/fatty_liver.pkl"
)

FIBROSIS_MODEL = (
    "/content/drive/MyDrive/"
    "Fibrosis Agent/"
    "XGBoost_model/"
    "xgboost_nafld.pkl"
)

CIRRHOSIS_MODEL = (
    "/content/drive/MyDrive/"
    ".Cirrhosis Agent/"
    "XGBoost_model/"
    "XGBoost_Cirrhosis.pkl"
)

TUMOR_MODEL = (
    "/content/drive/MyDrive/"
    "LiverAI/"
    "models/tumor/"
    "model.keras"
)

SEGMENTATION_MODEL = (
    "/content/drive/MyDrive/"
    "LiverAI/"
    "models/segmentation/"
    "model.pth"
)


# =============================================================================
# DATA PATHS
# =============================================================================

FATTY_ROOT = (
    "/content/drive/MyDrive/"
    "FattyLiver Agent"
)

FATTY_DATA = (
    "/content/drive/MyDrive/"
    "FattyLiver Agent/DATA"
)

FIBROSIS_ROOT = (
    "/content/drive/MyDrive/"
    "Fibrosis Agent"
)

CIRRHOSIS_ROOT = (
    "/content/drive/MyDrive/"
    ".Cirrhosis Agent"
)

CIRRHOSIS_DATA = (
    "/content/drive/MyDrive/"
    ".Cirrhosis Agent/"
    "DATA/liver_cirrhosis.csv"
)

TUMOR_ROOT = (
    "/content/drive/MyDrive/"
    "Liver CT Image Dataset"
)

SEGMENTATION_ROOT = (
    "/content/drive/MyDrive/"
    "archive (2)/image"
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

        "cirrhosis_data":
            CIRRHOSIS_DATA,

        "cirrhosis_model":
            CIRRHOSIS_MODEL,

        "tumor_root":
            TUMOR_ROOT,

        "tumor_model":
            TUMOR_MODEL,

        "segmentation_root":
            SEGMENTATION_ROOT,

        "segmentation_model":
            SEGMENTATION_MODEL,
    }

    results = {}

    for name, path in paths.items():

        exists = os.path.exists(
            path
        )

        results[name] = {

            "path":
                path,

            "exists":
                exists,
        }

        if exists:

            print(
                f"✓ {name}"
            )

            print(
                f"    {path}"
            )

        else:

            print(
                f"✗ {name}"
            )

            print(
                f"    {path}"
            )

    print("=" * 80)

    return results


# =============================================================================
# FIND MODELS
# =============================================================================

def find_model_files():

    print()
    print("=" * 80)
    print("SEARCHING FOR TRAINED MODEL FILES")
    print("=" * 80)

    search_roots = [

        "/content/drive/MyDrive/"
        "FattyLiver Agent",

        "/content/drive/MyDrive/"
        "Fibrosis Agent",

        "/content/drive/MyDrive/"
        ".Cirrhosis Agent",

        "/content/drive/MyDrive/"
        "LiverAI",

        "/content/drive/MyDrive/"
        "Liver CT Image Dataset",

        "/content/drive/MyDrive/"
        "archive (2)",
    ]

    extensions = [

        "*.pkl",
        "*.joblib",
        "*.keras",
        "*.h5",
        "*.pth",
        "*.pt",
        "*.onnx",
    ]

    found = []

    for root in search_roots:

        if not os.path.exists(
            root
        ):
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

                    found.append(
                        file
                    )

    found.sort()

    for file in found:

        print(file)

    print()
    print(
        "TOTAL MODEL FILES FOUND:",
        len(found)
    )

    print("=" * 80)

    return found


# =============================================================================
# LOAD PICKLE / JOBLIB
# =============================================================================

def load_pickle_model(path):

    if not os.path.exists(
        path
    ):

        raise FileNotFoundError(
            f"Model not found:\n{path}"
        )

    print(
        f"Loading model:\n{path}"
    )

    try:

        import joblib

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
        "✓ Model loaded"
    )

    print(
        "  Type:",
        type(model)
    )

    return model


# =============================================================================
# CREATE FATTY LIVER AGENT
# =============================================================================

def create_fatty_agent():

    if not os.path.exists(
        FATTY_MODEL
    ):

        print(
            "⚠ Fatty Liver model not found:"
        )

        print(
            FATTY_MODEL
        )

        return None

    try:

        from agents.fatty_liver_agent import (
            FattyLiverAgent
        )

        model = load_pickle_model(
            FATTY_MODEL
        )

        agent = FattyLiverAgent(
            model=model
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

    if not os.path.exists(
        FIBROSIS_MODEL
    ):

        print(
            "⚠ Fibrosis model not found:"
        )

        print(
            FIBROSIS_MODEL
        )

        return None

    try:

        from agents.fibrosis_agent import (
            FibrosisAgent
        )

        model = load_pickle_model(
            FIBROSIS_MODEL
        )

        print()
        print(
            "Fibrosis model information:"
        )

        if hasattr(
            model,
            "feature_names_in_"
        ):

            print(
                "FEATURES:",
                list(
                    model.feature_names_in_
                )
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

        agent = FibrosisAgent(
            model=model
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

    if not os.path.exists(
        CIRRHOSIS_MODEL
    ):

        print(
            "⚠ Cirrhosis model not found:"
        )

        print(
            CIRRHOSIS_MODEL
        )

        return None

    try:

        from agents.cirrhosis_agent import (
            CirrhosisAgent
        )

        package = load_pickle_model(
            CIRRHOSIS_MODEL
        )

        if not isinstance(
            package,
            dict
        ):

            raise TypeError(
                "Cirrhosis artifact must be a dictionary."
            )

        required_keys = [

            "model",

            "feature_names",

            "categorical_columns",

            "numerical_columns",
        ]

        missing = [

            key

            for key in required_keys

            if key not in package
        ]

        if missing:

            raise KeyError(
                "Missing cirrhosis keys: "
                f"{missing}"
            )

        agent = CirrhosisAgent(
            model_package=package
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

    if not os.path.exists(
        TUMOR_MODEL
    ):

        print(
            "⚠ Tumor model not found:"
        )

        print(
            TUMOR_MODEL
        )

        return None

    try:

        from agents.tumor_classification_agent import (
            TumorClassificationAgent
        )

        # IMPORTANT:
        # The agent itself loads the Keras model.
        # Do NOT load it a second time here.

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

    if not os.path.exists(
        SEGMENTATION_MODEL
    ):

        print(
            "⚠ Segmentation model not found:"
        )

        print(
            SEGMENTATION_MODEL
        )

        return None

    try:

        from agents.liver_segmentation_agent import (
            LiverSegmentationAgent
        )

        # IMPORTANT:
        # The agent loads the model itself.

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

    # IMPORTANT:
    # Argument names MUST match the orchestrator constructor.

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
        "Status:",
        result.get(
            "status"
        )
    )

    # -------------------------------------------------------------------------
    # AGENTS
    # -------------------------------------------------------------------------

    print()
    print(
        "AGENT RESULTS"
    )

    agents = result.get(
        "agents",
        {}
    )

    for agent_id, data in agents.items():

        if not isinstance(
            data,
            dict
        ):
            continue

        print()
        print(
            f"[{agent_id}]"
        )

        print(
            "  Task:",
            data.get(
                "task_type"
            )
        )

        print(
            "  Prediction:",
            data.get(
                "prediction"
            )
        )

        print(
            "  Confidence:",
            data.get(
                "confidence"
            )
        )

        print(
            "  Uncertainty:",
            data.get(
                "uncertainty"
            )
        )

        print(
            "  Trust:",
            data.get(
                "trust"
            )
        )

        print(
            "  Error:",
            data.get(
                "error"
            )
        )

    # -------------------------------------------------------------------------
    # ADAPTIVE FUSION
    # -------------------------------------------------------------------------

    print()
    print(
        "ADAPTIVE FUSION"
    )

    fusion = result.get(
        "adaptive_fusion",
        {}
    )

    print(
        "Status:",
        fusion.get(
            "status"
        )
    )

    print(
        "Weights:"
    )

    for agent_id, weight in (
        fusion.get(
            "weights",
            {}
        ).items()
    ):

        print(
            f"  {agent_id}: "
            f"{weight:.4f}"
        )

    # -------------------------------------------------------------------------
    # CONFLICTS
    # -------------------------------------------------------------------------

    print()
    print(
        "CONFLICTS:",
        len(
            result.get(
                "conflicts",
                []
            )
        )
    )

    # -------------------------------------------------------------------------
    # DECISION
    # -------------------------------------------------------------------------

    print()
    print(
        "DECISION"
    )

    decision = result.get(
        "decision",
        {}
    )

    print(
        "Status:",
        decision.get(
            "status"
        )
    )

    print(
        "Decision confidence:",
        decision.get(
            "decision_confidence"
        )
    )

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
        "Request additional tests:",
        decision.get(
            "request_additional_tests"
        )
    )

    print("=" * 80)


# =============================================================================
# EXAMPLE PIPELINE
# =============================================================================

def run_example(
    clinical_data=None,
    image=None,
    volume=None,
    patient_id="demo_patient",
):

    orchestrator = (
        create_orchestrator()
    )

    result = orchestrator.run(

        patient_id=
            patient_id,

        clinical_data=
            clinical_data,

        image=
            image,

        volume=
            volume,
    )

    print_final_result(
        result
    )

    return result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print()
    print(
        "=" * 80
    )

    print(
        "LIVERAI-MULTIAGENT"
    )

    print(
        "Adaptive Coordination Intelligence"
    )

    print(
        "=" * 80
    )

    # -------------------------------------------------------------------------
    # CHECK ENVIRONMENT
    # -------------------------------------------------------------------------

    check_paths()

    # -------------------------------------------------------------------------
    # MODEL INVENTORY
    # -------------------------------------------------------------------------

    print()
    find_model_files()

    # -------------------------------------------------------------------------
    # CREATE SYSTEM
    # -------------------------------------------------------------------------

    print()
    orchestrator = (
        create_orchestrator()
    )

    print()
    print(
        "=" * 80
    )

    print(
        "SYSTEM READY"
    )

    print(
        "Use:"
    )

    print(
        "orchestrator.run("
    )

    print(
        "    patient_id='patient_001',"
    )

    print(
        "    clinical_data=...,"
    )

    print(
        "    image=...,"
    )

    print(
        "    volume=..."
    )

    print(
        ")"
    )

    print(
        "=" * 80
    )
