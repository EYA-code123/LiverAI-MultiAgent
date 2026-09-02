# =============================================================================
# LiverAI-MultiAgent
# MAIN ENTRY POINT
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

os.chdir(
    PROJECT_DIR
)


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
# CHECK PATHS
# =============================================================================

def check_paths():

    paths = {

        "fatty":
            FATTY_MODEL,

        "fibrosis":
            FIBROSIS_MODEL,

        "cirrhosis":
            CIRRHOSIS_MODEL,

        "tumor":
            TUMOR_MODEL,

        "segmentation":
            SEGMENTATION_MODEL
    }

    print(
        "=" * 80
    )

    print(
        "CHECKING MODEL PATHS"
    )

    print(
        "=" * 80
    )

    results = {}

    for name, path in paths.items():

        exists = os.path.exists(
            path
        )

        results[name] = exists

        if exists:

            print(
                f"✓ {name:<15}"
                f"{path}"
            )

        else:

            print(
                f"✗ {name:<15}"
                f"{path}"
            )

    print(
        "=" * 80
    )

    return results


# =============================================================================
# LOAD PICKLE / JOBLIB
# =============================================================================

def load_pickle_model(
    path
):

    if not os.path.exists(
        path
    ):

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

    return model


# =============================================================================
# CREATE FATTY AGENT
# =============================================================================

def create_fatty_agent():

    try:

        from agents.fatty_liver_agent import (
            FattyLiverAgent
        )

        package = (
            load_pickle_model(
                FATTY_MODEL
            )
        )

        return FattyLiverAgent(
            model_package=package
        )

    except Exception as e:

        print(
            "✗ Fatty Liver Agent:",
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

        model = (
            load_pickle_model(
                FIBROSIS_MODEL
            )
        )

        return FibrosisAgent(
            model=model
        )

    except Exception as e:

        print(
            "✗ Fibrosis Agent:",
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

        package = (
            load_pickle_model(
                CIRRHOSIS_MODEL
            )
        )

        if not isinstance(
            package,
            dict
        ):

            raise TypeError(
                "Cirrhosis model must "
                "be a dictionary package."
            )

        required_keys = [

            "model",

            "feature_names",

            "categorical_columns",

            "numerical_columns"
        ]

        missing = [

            key

            for key
            in required_keys

            if key
            not in package
        ]

        if missing:

            raise KeyError(
                "Missing cirrhosis "
                f"keys: {missing}"
            )

        return CirrhosisAgent(
            model_package=package
        )

    except Exception as e:

        print(
            "✗ Cirrhosis Agent:",
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

        return TumorClassificationAgent(
            model_path=TUMOR_MODEL
        )

    except Exception as e:

        print(
            "✗ Tumor Agent:",
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

        return LiverSegmentationAgent(
            model_path=
                SEGMENTATION_MODEL
        )

    except Exception as e:

        print(
            "✗ Segmentation Agent:",
            e
        )

        traceback.print_exc()

        return None


# =============================================================================
# CREATE CLINICAL REASONING AGENT
# =============================================================================

def create_clinical_reasoning_agent():

    try:

        from agents.clinical_reasoning_agent import (
            ClinicalReasoningAgent
        )

        # The current repository version
        # expects a model package.

        print(
            "⚠ Clinical reasoning "
            "agent requires its saved "
            "model package."
        )

        return None

    except Exception as e:

        print(
            "✗ Clinical Reasoning Agent:",
            e
        )

        traceback.print_exc()

        return None


# =============================================================================
# CREATE ALL AGENTS
# =============================================================================

def create_agents():

    agents = {

        "fatty_liver":
            create_fatty_agent(),

        "fibrosis":
            create_fibrosis_agent(),

        "cirrhosis":
            create_cirrhosis_agent(),

        "tumor":
            create_tumor_agent(),

        "segmentation":
            create_segmentation_agent(),

        "clinical_reasoning":
            create_clinical_reasoning_agent()
    }

    print(
        "\n"
        + "=" * 80
    )

    print(
        "AGENT INITIALIZATION"
    )

    print(
        "=" * 80
    )

    for name, agent in agents.items():

        if agent is None:

            print(
                f"✗ {name}"
            )

        else:

            print(
                f"✓ {name}"
            )

    print(
        "=" * 80
    )

    return agents


# =============================================================================
# CREATE ORCHESTRATOR
# =============================================================================

def create_orchestrator():

    from orchestrator.liver_orchestrator import (
        LiverAIOrchestrator
    )

    agents = create_agents()

    orchestrator = (
        LiverAIOrchestrator(

            fatty_liver_agent=
                agents["fatty_liver"],

            fibrosis_agent=
                agents["fibrosis"],

            cirrhosis_agent=
                agents["cirrhosis"],

            tumor_agent=
                agents["tumor"],

            segmentation_agent=
                agents["segmentation"],

            clinical_agent=
                agents["clinical_reasoning"]
        )
    )

    return orchestrator


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "=" * 80
    )

    print(
        "LIVERAI MULTI-AGENT SYSTEM"
    )

    print(
        "=" * 80
    )

    check_paths()

    orchestrator = (
        create_orchestrator()
    )

    print(
        "\nLiverAI orchestrator ready."
    )

    return orchestrator


if __name__ == "__main__":

    orchestrator = main()
