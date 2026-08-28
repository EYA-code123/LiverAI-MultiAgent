# =============================================================================
# LiverAI-MultiAgent
# MAIN.PY - VERSION CORRIGÉE POUR GOOGLE COLAB
# =============================================================================

import os
import sys
import pickle
import joblib
import traceback
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = "/content/LiverAI-MultiAgent"

DRIVE_BASE = "/content/drive/MyDrive"

LIVERAI_DIR = os.path.join(
    DRIVE_BASE,
    "LiverAI"
)


# =============================================================================
# GOOGLE DRIVE - REAL PROJECT PATHS
# =============================================================================

FATTY_ROOT = os.path.join(
    DRIVE_BASE,
    "FattyLiver Agent"
)

FATTY_DATA = os.path.join(
    FATTY_ROOT,
    "DATA"
)


FIBROSIS_ROOT = os.path.join(
    DRIVE_BASE,
    "Fibrosis Agent"
)

FIBROSIS_MODEL_PATH = os.path.join(
    FIBROSIS_ROOT,
    "XGBoost_model",
    "xgboost_nafld.pkl"
)


CIRRHOSIS_ROOT = os.path.join(
    DRIVE_BASE,
    ".Cirrhosis Agent"
)

CIRRHOSIS_DATA = os.path.join(
    CIRRHOSIS_ROOT,
    "DATA",
    "liver_cirrhosis.csv"
)


TUMOR_ROOT = os.path.join(
    DRIVE_BASE,
    "Liver CT Image Dataset"
)


SEGMENTATION_ROOT = os.path.join(
    DRIVE_BASE,
    "archive (2)"
)


# =============================================================================
# POSSIBLE MODEL DIRECTORIES
# =============================================================================

MODELS_DIR = os.path.join(
    LIVERAI_DIR,
    "models"
)

DATASETS_DIR = os.path.join(
    LIVERAI_DIR,
    "datasets"
)


FATTY_MODEL_DIR = os.path.join(
    MODELS_DIR,
    "fatty_liver"
)

FIBROSIS_MODEL_DIR = os.path.join(
    MODELS_DIR,
    "fibrosis"
)

CIRRHOSIS_MODEL_DIR = os.path.join(
    MODELS_DIR,
    "cirrhosis"
)

TUMOR_MODEL_DIR = os.path.join(
    MODELS_DIR,
    "tumor"
)

SEGMENTATION_MODEL_DIR = os.path.join(
    MODELS_DIR,
    "segmentation"
)


# =============================================================================
# POSSIBLE MODEL FILES
# =============================================================================

FATTY_MODEL_CANDIDATES = [

    os.path.join(
        FATTY_MODEL_DIR,
        "model.pkl"
    ),

    os.path.join(
        FATTY_ROOT,
        "models",
        "model.pkl"
    ),

    os.path.join(
        FATTY_ROOT,
        "models_RF",
        "model.pkl"
    ),

    os.path.join(
        FATTY_ROOT,
        "models_LGBM",
        "model.pkl"
    ),

    os.path.join(
        FATTY_ROOT,
        "model.pkl"
    ),

    os.path.join(
        FATTY_ROOT,
        "random_forest.pkl"
    ),

    os.path.join(
        FATTY_ROOT,
        "lightgbm.pkl"
    )
]


CIRRHOSIS_MODEL_CANDIDATES = [

    os.path.join(
        CIRRHOSIS_MODEL_DIR,
        "XGBoost_Cirrhosis.pkl"
    ),

    os.path.join(
        CIRRHOSIS_ROOT,
        "XGBoost_Cirrhosis.pkl"
    ),

    os.path.join(
        CIRRHOSIS_ROOT,
        "RandomForest_Cirrhosis.pkl"
    ),

    os.path.join(
        CIRRHOSIS_ROOT,
        "models",
        "XGBoost_Cirrhosis.pkl"
    )
]


TUMOR_MODEL_CANDIDATES = [

    os.path.join(
        MODELS_DIR,
        "tumor",
        "model.keras"
    ),

    os.path.join(
        TUMOR_ROOT,
        "model.keras"
    ),

    os.path.join(
        TUMOR_ROOT,
        "model.h5"
    ),

    os.path.join(
        TUMOR_ROOT,
        "tumor_model.keras"
    ),

    os.path.join(
        TUMOR_ROOT,
        "tumor_model.h5"
    )
]


SEGMENTATION_MODEL_CANDIDATES = [

    os.path.join(
        SEGMENTATION_MODEL_DIR,
        "model.keras"
    ),

    os.path.join(
        SEGMENTATION_ROOT,
        "model.keras"
    ),

    os.path.join(
        SEGMENTATION_ROOT,
        "model.h5"
    ),

    os.path.join(
        SEGMENTATION_ROOT,
        "segmentation_model.keras"
    ),

    os.path.join(
        SEGMENTATION_ROOT,
        "segmentation_model.h5"
    )
]


# =============================================================================
# PATH DICTIONARY
# =============================================================================

PATHS = {

    "project": PROJECT_DIR,

    "liverai": LIVERAI_DIR,

    "fatty_root": FATTY_ROOT,

    "fatty_data": FATTY_DATA,

    "fibrosis_root": FIBROSIS_ROOT,

    "fibrosis_model": FIBROSIS_MODEL_PATH,

    "cirrhosis_root": CIRRHOSIS_ROOT,

    "cirrhosis_data": CIRRHOSIS_DATA,

    "tumor_root": TUMOR_ROOT,

    "segmentation_root": SEGMENTATION_ROOT
}


# =============================================================================
# PROJECT INITIALIZATION
# =============================================================================

if PROJECT_DIR not in sys.path:

    sys.path.insert(
        0,
        PROJECT_DIR
    )


if os.path.isdir(PROJECT_DIR):

    os.chdir(
        PROJECT_DIR
    )


print("=" * 80)
print("LIVERAI MULTI-AGENT SYSTEM")
print("=" * 80)

print("PROJECT:")
print(PROJECT_DIR)

print("GOOGLE DRIVE:")
print(DRIVE_BASE)

print("=" * 80)


# =============================================================================
# CHECK PATHS
# =============================================================================

def check_paths():

    print()
    print("=" * 80)
    print("LIVERAI DATASET / MODEL PATH CHECK")
    print("=" * 80)

    results = {}

    for name, path in PATHS.items():

        exists = os.path.exists(path)

        results[name] = {
            "path": path,
            "exists": exists
        }

        if exists:

            print(
                f"✓ {name:<20}"
            )

            print(
                f"  {path}"
            )

        else:

            print(
                f"✗ {name:<20}"
            )

            print(
                f"  {path}"
            )

    print("=" * 80)

    return results


# =============================================================================
# FIND MODEL FILES
# =============================================================================

def find_model_files(
    search_roots=None
):

    print()
    print("=" * 80)
    print("SEARCHING FOR TRAINED MODEL FILES")
    print("=" * 80)

    if search_roots is None:

        search_roots = [

            FATTY_ROOT,

            FIBROSIS_ROOT,

            CIRRHOSIS_ROOT,

            TUMOR_ROOT,

            SEGMENTATION_ROOT,

            LIVERAI_DIR
        ]


    extensions = {

        ".pkl",
        ".pickle",
        ".joblib",
        ".keras",
        ".h5",
        ".hdf5",
        ".pt",
        ".pth",
        ".onnx"
    }


    found = []


    for root in search_roots:

        if not os.path.exists(root):

            continue


        for current_root, dirs, files in os.walk(root):

            # Ne pas parcourir les caches inutiles
            dirs[:] = [

                d for d in dirs

                if d not in {

                    ".git",
                    "__pycache__",
                    ".ipynb_checkpoints"
                }
            ]


            for filename in files:

                extension = Path(
                    filename
                ).suffix.lower()


                if extension in extensions:

                    full_path = os.path.join(
                        current_root,
                        filename
                    )

                    found.append(
                        full_path
                    )


    # Supprimer doublons
    found = sorted(
        list(set(found))
    )


    if len(found) == 0:

        print(
            "No trained model files found."
        )

    else:

        for path in found:

            print(path)


    print()
    print("=" * 80)

    print(
        f"TOTAL MODEL FILES FOUND: {len(found)}"
    )

    print("=" * 80)

    return found


# =============================================================================
# SHOW MODEL INVENTORY
# =============================================================================

def show_model_inventory():

    print()
    print("=" * 80)
    print("LIVERAI MODEL INVENTORY")
    print("=" * 80)


    inventory = {

        "Fatty Liver": FATTY_MODEL_CANDIDATES,

        "Fibrosis": [
            FIBROSIS_MODEL_PATH
        ],

        "Cirrhosis": CIRRHOSIS_MODEL_CANDIDATES,

        "Tumor": TUMOR_MODEL_CANDIDATES,

        "Segmentation": SEGMENTATION_MODEL_CANDIDATES
    }


    result = {}


    for agent_name, candidates in inventory.items():

        print()
        print(
            agent_name.upper()
        )

        print("-" * 60)


        existing = []


        for path in candidates:

            if os.path.isfile(path):

                existing.append(
                    path
                )

                print(
                    f"✓ {path}"
                )


        if len(existing) == 0:

            print(
                "✗ No trained model found"
            )


        result[agent_name] = existing


    print()
    print("=" * 80)

    return result


# =============================================================================
# GENERIC MODEL SEARCH
# =============================================================================

def find_first_existing(
    candidates
):

    for path in candidates:

        if os.path.isfile(path):

            return path


    return None


# =============================================================================
# LOAD PICKLE / JOBLIB MODEL
# =============================================================================

def load_pickle_model(path):

    if not path:

        raise FileNotFoundError(
            "No model path was provided."
        )


    if not os.path.isfile(path):

        raise FileNotFoundError(
            f"Model not found:\n{path}"
        )


    print()
    print(
        "Loading model:"
    )

    print(
        path
    )


    try:

        model = joblib.load(
            path
        )

    except Exception as joblib_error:

        print(
            "joblib.load failed."
        )

        print(
            "Trying pickle..."
        )

        try:

            with open(
                path,
                "rb"
            ) as f:

                model = pickle.load(
                    f
                )

        except Exception as pickle_error:

            raise RuntimeError(
                "Unable to load model.\n"
                f"joblib error: {joblib_error}\n"
                f"pickle error: {pickle_error}"
            )


    print(
        "✓ Model loaded"
    )

    print(
        "Type:",
        type(model)
    )


    return model


# =============================================================================
# LOAD KERAS MODEL
# =============================================================================

def load_keras_model(path):

    if not path:

        raise FileNotFoundError(
            "No Keras model path was provided."
        )


    if not os.path.isfile(path):

        raise FileNotFoundError(
            f"Keras model not found:\n{path}"
        )


    import tensorflow as tf


    print()
    print(
        "Loading Keras model:"
    )

    print(
        path
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
# FAT VISIBILITY
# =============================================================================

def create_fatty_agent():

    print()
    print("-" * 80)
    print("FATTY LIVER AGENT")
    print("-" * 80)


    path = find_first_existing(
        FATTY_MODEL_CANDIDATES
    )


    if path is None:

        print(
            "⚠ No trained Fatty Liver model found."
        )

        print(
            "Fatty Liver Agent cannot be initialized yet."
        )

        return None


    model = load_pickle_model(
        path
    )


    agent = FattyLiverAgent(
        model=model
    )


    print(
        "✓ Fatty Liver Agent initialized"
    )


    return agent


# =============================================================================
# FIBROSIS
# =============================================================================

def create_fibrosis_agent():

    print()
    print("-" * 80)
    print("FIBROSIS AGENT")
    print("-" * 80)


    model = load_pickle_model(
        FIBROSIS_MODEL_PATH
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


# =============================================================================
# CIRRHOSIS
# =============================================================================

def create_cirrhosis_agent():

    print()
    print("-" * 80)
    print("CIRRHOSIS AGENT")
    print("-" * 80)


    path = find_first_existing(
        CIRRHOSIS_MODEL_CANDIDATES
    )


    if path is None:

        print(
            "⚠ No trained Cirrhosis model found."
        )

        print(
            "Cirrhosis Agent cannot be initialized yet."
        )

        return None


    package = load_pickle_model(
        path
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


    # Le notebook XGBoost sauvegarde actuellement :
    #
    # model
    # feature_names
    # categorical_columns
    # numerical_columns
    # encoders
    # target_encoder
    #
    # Donc on accepte cette structure.


    required_keys = [

        "model",

        "feature_names",

        "categorical_columns",

        "numerical_columns",

        "encoders",

        "target_encoder"
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
# TUMOR
# =============================================================================

def create_tumor_agent():

    print()
    print("-" * 80)
    print("TUMOR CLASSIFICATION AGENT")
    print("-" * 80)


    path = find_first_existing(
        TUMOR_MODEL_CANDIDATES
    )


    if path is None:

        print(
            "⚠ No trained Tumor model found."
        )

        print(
            "Tumor Agent cannot be initialized yet."
        )

        return None


    # Le constructeur de ton agent demande model_path
    agent = TumorClassificationAgent(
        model_path=path
    )


    print(
        "✓ Tumor Classification Agent initialized"
    )


    return agent


# =============================================================================
# SEGMENTATION
# =============================================================================

def create_segmentation_agent():

    print()
    print("-" * 80)
    print("LIVER SEGMENTATION AGENT")
    print("-" * 80)


    path = find_first_existing(
        SEGMENTATION_MODEL_CANDIDATES
    )


    if path is None:

        print(
            "⚠ No trained Segmentation model found."
        )

        print(
            "Segmentation Agent cannot be initialized yet."
        )

        return None


    agent = LiverSegmentationAgent(
        model_path=path
    )


    print(
        "✓ Liver Segmentation Agent initialized"
    )


    return agent


# =============================================================================
# CLINICAL REASONING
# =============================================================================

def create_clinical_reasoning_agent():

    print()
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

    print()
    print("=" * 80)
    print("INITIALIZING ALL LIVERAI AGENTS")
    print("=" * 80)


    agents = {

        "fatty_liver": None,

        "fibrosis": None,

        "cirrhosis": None,

        "tumor": None,

        "segmentation": None,

        "clinical_reasoning": None
    }


    # -------------------------------------------------------------------------
    # FAT
    # -------------------------------------------------------------------------

    try:

        agents["fatty_liver"] = (
            create_fatty_agent()
        )

    except Exception as e:

        print(
            f"✗ Fatty Liver Agent failed: {e}"
        )

        traceback.print_exc()


    # -------------------------------------------------------------------------
    # FIBROSIS
    # -------------------------------------------------------------------------

    try:

        agents["fibrosis"] = (
            create_fibrosis_agent()
        )

    except Exception as e:

        print(
            f"✗ Fibrosis Agent failed: {e}"
        )

        traceback.print_exc()


    # -------------------------------------------------------------------------
    # CIRRHOSIS
    # -------------------------------------------------------------------------

    try:

        agents["cirrhosis"] = (
            create_cirrhosis_agent()
        )

    except Exception as e:

        print(
            f"✗ Cirrhosis Agent failed: {e}"
        )

        traceback.print_exc()


    # -------------------------------------------------------------------------
    # TUMOR
    # -------------------------------------------------------------------------

    try:

        agents["tumor"] = (
            create_tumor_agent()
        )

    except Exception as e:

        print(
            f"✗ Tumor Agent failed: {e}"
        )

        traceback.print_exc()


    # -------------------------------------------------------------------------
    # SEGMENTATION
    # -------------------------------------------------------------------------

    try:

        agents["segmentation"] = (
            create_segmentation_agent()
        )

    except Exception as e:

        print(
            f"✗ Segmentation Agent failed: {e}"
        )

        traceback.print_exc()


    # -------------------------------------------------------------------------
    # CLINICAL
    # -------------------------------------------------------------------------

    try:

        agents["clinical_reasoning"] = (
            create_clinical_reasoning_agent()
        )

    except Exception as e:

        print(
            f"✗ Clinical Reasoning Agent failed: {e}"
        )

        traceback.print_exc()


    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("INITIALIZATION SUMMARY")
    print("=" * 80)


    initialized = 0


    for name, agent in agents.items():

        if agent is not None:

            print(
                f"✓ {name}"
            )

            initialized += 1

        else:

            print(
                f"✗ {name}"
            )


    print("=" * 80)

    print(
        f"Initialized: {initialized}/{len(agents)}"
    )

    print("=" * 80)


    return agents


# =============================================================================
# CREATE ORCHESTRATOR
# =============================================================================

def create_orchestrator():

    print()
    print("=" * 80)
    print("CREATING LIVERAI ORCHESTRATOR")
    print("=" * 80)


    agents = create_agents()


    orchestrator = LiverAIOrchestrator(

        fatty_agent=agents["fatty_liver"],

        fibrosis_agent=agents["fibrosis"],

        cirrhosis_agent=agents["cirrhosis"],

        tumor_agent=agents["tumor"],

        segmentation_agent=agents["segmentation"],

        clinical_reasoning_agent=agents[
            "clinical_reasoning"
        ]
    )


    print()
    print("=" * 80)
    print("✓ LIVERAI ORCHESTRATOR CREATED")
    print("=" * 80)


    return orchestrator


# =============================================================================
# ARCHITECTURE
# =============================================================================

def show_architecture():

    print(
        """
===============================================================================

                         LIVERAI MULTI-AGENT SYSTEM

                              PATIENT DATA
                                   |
                                   v
                     +---------------------------+
                     |    LIVERAI ORCHESTRATOR   |
                     +-------------+-------------+
                                   |
             +----------+----------+----------+----------+
             |          |          |          |          |
             v          v          v          v          v
          FAT/NAFLD  FIBROSIS  CIRRHOSIS   TUMOR   SEGMENTATION
             |          |          |          |          |
             +----------+----------+----------+----------+
                                   |
                                   v
                     +---------------------------+
                     | CLINICAL REASONING AGENT  |
                     +-------------+-------------+
                                   |
                                   v
                         UNIFIED LIVER ASSESSMENT

===============================================================================
"""
    )


# =============================================================================
# SYSTEM STATUS
# =============================================================================

def system_status(orchestrator):

    print()
    print("=" * 80)
    print("SYSTEM STATUS")
    print("=" * 80)


    if hasattr(
        orchestrator,
        "agents"
    ):

        for name, agent in orchestrator.agents.items():

            if agent is None:

                print(
                    f"✗ {name:<25} NOT INITIALIZED"
                )

            else:

                print(
                    f"✓ {name:<25} READY"
                )

    else:

        print(
            "⚠ Orchestrator has no 'agents' attribute."
        )


    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print("=" * 80)
    print("STARTING LIVERAI")
    print("=" * 80)


    check_paths()


    show_model_inventory()


    show_architecture()


    orchestrator = create_orchestrator()


    system_status(
        orchestrator
    )


    return orchestrator


# =============================================================================
# IMPORT AGENTS
#
# Placé ici pour que les fonctions de diagnostic puissent être importées
# même si un agent optionnel rencontre un problème de dépendance.
# =============================================================================

try:

    from agents.fatty_liver_agent import (
        FattyLiverAgent
    )

except Exception as e:

    print(
        "WARNING: FattyLiverAgent import failed:",
        e
    )

    FattyLiverAgent = None


try:

    from agents.fibrosis_agent import (
        FibrosisAgent
    )

except Exception as e:

    print(
        "WARNING: FibrosisAgent import failed:",
        e
    )

    FibrosisAgent = None


try:

    from agents.cirrhosis_agent import (
        CirrhosisAgent
    )

except Exception as e:

    print(
        "WARNING: CirrhosisAgent import failed:",
        e
    )

    CirrhosisAgent = None


try:

    from agents.tumor_classification_agent import (
        TumorClassificationAgent
    )

except Exception as e:

    print(
        "WARNING: TumorClassificationAgent import failed:",
        e
    )

    TumorClassificationAgent = None


try:

    from agents.liver_segmentation_agent import (
        LiverSegmentationAgent
    )

except Exception as e:

    print(
        "WARNING: LiverSegmentationAgent import failed:",
        e
    )

    LiverSegmentationAgent = None


try:

    from agents.clinical_reasoning_agent import (
        ClinicalReasoningAgent
    )

except Exception as e:

    print(
        "WARNING: ClinicalReasoningAgent import failed:",
        e
    )

    ClinicalReasoningAgent = None


# =============================================================================
# ORCHESTRATOR IMPORT
# =============================================================================

try:

    from orchestrator.liver_orchestrator import (
        LiverAIOrchestrator
    )

except Exception as e:

    print(
        "WARNING: LiverAIOrchestrator import failed:",
        e
    )

    LiverAIOrchestrator = None


# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":

    orchestrator = main()
