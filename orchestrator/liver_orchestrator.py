def __init__(self):

    self.name = "LiverAI Multi-Agent Orchestrator"

    # =========================================================================
    # MODEL PATHS
    # =========================================================================

    FATTY_MODEL_PATH = (
        "/content/drive/MyDrive/"
        "Fatty_Liver_Dataset/models/FattyLiver_LightGBM.pkl"
    )

    FIBROSIS_MODEL_PATH = (
        "/content/drive/MyDrive/"
        "Fibrosis Agent/XGBoost_model/xgboost_nafld.pkl"
    )

    CIRRHOSIS_MODEL_PATH = (
        "/content/drive/MyDrive/"
        ".Cirrhosis Agent/XGBoost_model/"
        "XGBoost_Cirrhosis_fixed.joblib"
    )

    TUMOR_MODEL_PATH = (
        "/content/drive/MyDrive/"
        "models/tumor/efficientnet_b0_best.pth"
    )

    SEGMENTATION_MODEL_PATH = (
        "/content/drive/MyDrive/"
        "Liver Segmentation Agent/models/"
        "SegResNet3D_Liver_best.pth"
    )

    CLINICAL_MODEL_PATH = (
        "/content/drive/MyDrive/"
        "Clinical Reasoning Agent/tabtransformer_bupa"
    )

    # =========================================================================
    # INITIALIZE AGENTS
    # =========================================================================

    self.cirrhosis_agent = None
    self.fatty_liver_agent = None
    self.fibrosis_agent = None
    self.tumor_agent = None
    self.segmentation_agent = None
    self.clinical_agent = None

    # -------------------------------------------------------------------------
    # FATIGUE / FATTY LIVER
    # -------------------------------------------------------------------------

    try:
        import joblib
        from agents.fatty_liver_agent import FattyLiverAgent

        fatty_model = joblib.load(FATTY_MODEL_PATH)

        self.fatty_liver_agent = FattyLiverAgent(
            fatty_model
        )

        print("✓ Fatty Liver Agent loaded")

    except Exception as e:
        print("✗ Fatty Liver Agent failed:")
        print(f"  {type(e).__name__}: {e}")

    # -------------------------------------------------------------------------
    # FIBROSIS
    # -------------------------------------------------------------------------

    try:
        import joblib
        from agents.fibrosis_agent import FibrosisAgent

        fibrosis_model = joblib.load(
            FIBROSIS_MODEL_PATH
        )

        self.fibrosis_agent = FibrosisAgent(
            fibrosis_model
        )

        print("✓ Fibrosis Agent loaded")

    except Exception as e:
        print("✗ Fibrosis Agent failed:")
        print(f"  {type(e).__name__}: {e}")

    # -------------------------------------------------------------------------
    # CIRRHOSIS
    # -------------------------------------------------------------------------

    try:
        from agents.cirrhosis_agent import CirrhosisAgent

        self.cirrhosis_agent = CirrhosisAgent(
            CIRRHOSIS_MODEL_PATH
        )

        print("✓ Cirrhosis Agent loaded")

    except Exception as e:
        print("✗ Cirrhosis Agent failed:")
        print(f"  {type(e).__name__}: {e}")

    # -------------------------------------------------------------------------
    # TUMOR CLASSIFICATION
    # -------------------------------------------------------------------------

    try:
        from agents.tumor_classification_agent import (
            TumorClassificationAgent
        )

        self.tumor_agent = TumorClassificationAgent(
            TUMOR_MODEL_PATH
        )

        print("✓ Tumor Classification Agent loaded")

    except Exception as e:
        print("✗ Tumor Classification Agent failed:")
        print(f"  {type(e).__name__}: {e}")

    # -------------------------------------------------------------------------
    # LIVER SEGMENTATION
    # -------------------------------------------------------------------------

    try:
        from agents.liver_segmentation_agent import (
            LiverSegmentationAgent
        )

        self.segmentation_agent = LiverSegmentationAgent(
            model_path=SEGMENTATION_MODEL_PATH
        )

        print("✓ Liver Segmentation Agent loaded")

    except Exception as e:
        print("✗ Liver Segmentation Agent failed:")
        print(f"  {type(e).__name__}: {e}")

    # -------------------------------------------------------------------------
    # CLINICAL REASONING
    # -------------------------------------------------------------------------

    try:
        from agents.clinical_reasoning_agent import (
            ClinicalReasoningAgent
        )

        self.clinical_agent = ClinicalReasoningAgent(
            CLINICAL_MODEL_PATH
        )

        print("✓ Clinical Reasoning Agent loaded")

    except Exception as e:
        print("✗ Clinical Reasoning Agent failed:")
        print(f"  {type(e).__name__}: {e}")

    # =========================================================================
    # AGENT REGISTRY
    # =========================================================================

    self.agents = {

        "fatty_liver":
            self.fatty_liver_agent,

        "fibrosis":
            self.fibrosis_agent,

        "cirrhosis":
            self.cirrhosis_agent,

        "tumor_classification":
            self.tumor_agent,

        "liver_segmentation":
            self.segmentation_agent,

        "clinical_reasoning":
            self.clinical_agent,
    }

    # =========================================================================
    # COORDINATION COMPONENTS
    # =========================================================================

    self.trust_manager = (
        TrustManager()
        if TrustManager is not None
        else None
    )

    self.adaptive_fusion = (
        AdaptiveFusion()
        if AdaptiveFusion is not None
        else None
    )

    self.conflict_detector = (
        ConflictDetector()
        if ConflictDetector is not None
        else None
    )

    self.decision_engine = (
        DecisionEngine()
        if DecisionEngine is not None
        else None
    )

    # =========================================================================
    # STATE
    # =========================================================================

    self.last_results = {}
    self.last_assessment = None
    self.execution_log = []

    # =========================================================================
    # STATUS
    # =========================================================================

    print("=" * 80)
    print("LIVERAI MULTI-AGENT ORCHESTRATOR")
    print("=" * 80)

    print("\nRegistered Agents:")

    for name, agent in self.agents.items():

        status = (
            "READY"
            if agent is not None
            else "NOT LOADED"
        )

        print(
            f"  {name:<25} : {status}"
        )

    print("=" * 80)
