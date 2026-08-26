import os
import json
import traceback
import joblib
import pickle

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
# LOAD MODEL HELPERS
# ==============================================================

def load_pickle_model(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found: {path}"
        )

    print(f"Loading model: {path}")

    try:
        model = joblib.load(path)

    except Exception:

        with open(path, "rb") as f:
            model = pickle.load(f)

    print("✓ Model loaded")

    return model


def load_keras_model(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found: {path}"
        )

    import tensorflow as tf

    print(f"Loading Keras model: {path}")

    model = tf.keras.models.load_model(
        path,
        compile=False
    )

    print("✓ Keras model loaded")

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
    # 1. FATTY LIVER
    # ==========================================================

    print("\n[1/6] Initializing Fatty Liver Agent...")

    fatty_model = load_pickle_model(
        FATTY_MODEL_PATH
    )

    fatty_liver_agent = FattyLiverAgent(
        model=fatty_model
    )

    print("✓ Fatty Liver Agent ready")

    # ==========================================================
    # 2. FIBROSIS
    # ==========================================================

    print("\n[2/6] Initializing Fibrosis Agent...")

    fibrosis_model = load_pickle_model(
        FIBROSIS_MODEL_PATH
    )

    fibrosis_agent = FibrosisAgent(
        model=fibrosis_model
    )

    print("✓ Fibrosis Agent ready")

    # ==========================================================
    # 3. CIRRHOSIS
    # ==========================================================

    print("\n[3/6] Initializing Cirrhosis Agent...")

    cirrhosis_package = load_pickle_model(
        CIRRHOSIS_MODEL_PATH
    )

    cirrhosis_agent = CirrhosisAgent(
        model_package=cirrhosis_package
    )

    print("✓ Cirrhosis Agent ready")

    # ==========================================================
    # 4. TUMOR CLASSIFICATION
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
    # 5. LIVER SEGMENTATION
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
    # 6. CLINICAL REASONING
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

    orchestrator = LiverAIOrchestrator(

        fatty_agent=fatty_liver_agent,

        fibrosis_agent=fibrosis_agent,

        cirrhosis_agent=cirrhosis_agent,

        tumor_agent=tumor_classification_agent,

        segmentation_agent=liver_segmentation_agent,

        clinical_reasoning_agent=clinical_reasoning_agent
    )

    print("\n✓ LiverAI Orchestrator created")

    return orchestrator
