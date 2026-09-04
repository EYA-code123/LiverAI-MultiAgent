def __init__(
    self,
    cirrhosis_agent=None,
    fatty_liver_agent=None,
    clinical_agent=None,
    fibrosis_agent=None,
    tumor_agent=None,
    segmentation_agent=None,
):

    self.name = "LiverAI Multi-Agent Orchestrator"

    # =========================================================================
    # MODEL PATHS
    # =========================================================================

    DEFAULT_MODEL_PATHS = {

        "fatty_liver":
            "/content/drive/MyDrive/Fatty_Liver_Dataset/models/FattyLiver_LightGBM.pkl",

        "fibrosis":
            "/content/drive/MyDrive/Fibrosis Agent/XGBoost_model/xgboost_nafld.pkl",

        "cirrhosis":
            "/content/drive/MyDrive/.Cirrhosis Agent/XGBoost_model/XGBoost_Cirrhosis_fixed.joblib",

        "tumor":
            "/content/drive/MyDrive/models/tumor/efficientnet_b0_best.pth",

        "segmentation":
            "/content/drive/MyDrive/Liver Segmentation Agent/models/SegResNet3D_Liver_best.pth",

        "clinical":
            "/content/drive/MyDrive/Clinical Reasoning Agent/tabtransformer_bupa",
    }

    # =========================================================================
    # IMPORTS
    # =========================================================================

    import os
    import joblib

    from agents.fatty_liver_agent import FattyLiverAgent
    from agents.fibrosis_agent import FibrosisAgent
    from agents.cirrhosis_agent import CirrhosisAgent
    from agents.tumor_classification_agent import TumorClassificationAgent
    from agents.liver_segmentation_agent import LiverSegmentationAgent
    from agents.clinical_reasoning_agent import ClinicalReasoningAgent

    # =========================================================================
    # FATY LIVER
    # =========================================================================

    if fatty_liver_agent is None:

        path = DEFAULT_MODEL_PATHS["fatty_liver"]

        try:

            print("\nLoading Fatty Liver Agent...")

            if not os.path.exists(path):
                raise FileNotFoundError(path)

            model_package = joblib.load(path)

            fatty_liver_agent = FattyLiverAgent(
                model_package
            )

            print("✓ Fatty Liver Agent loaded")

        except Exception as e:

            print(f"✗ Fatty Liver Agent failed: {e}")

    # =========================================================================
    # FIBROSIS
    # =========================================================================

    if fibrosis_agent is None:

        path = DEFAULT_MODEL_PATHS["fibrosis"]

        try:

            print("\nLoading Fibrosis Agent...")

            if not os.path.exists(path):
                raise FileNotFoundError(path)

            model = joblib.load(path)

            fibrosis_agent = FibrosisAgent(
                model
            )

            print("✓ Fibrosis Agent loaded")

        except Exception as e:

            print(f"✗ Fibrosis Agent failed: {e}")

    # =========================================================================
    # CIRRHOSIS
    # =========================================================================

    if cirrhosis_agent is None:

        path = DEFAULT_MODEL_PATHS["cirrhosis"]

        try:

            print("\nLoading Cirrhosis Agent...")

            if not os.path.exists(path):
                raise FileNotFoundError(path)

            cirrhosis_agent = CirrhosisAgent(
                path
            )

            print("✓ Cirrhosis Agent loaded")

        except Exception as e:

            print(f"✗ Cirrhosis Agent failed: {e}")

    # =========================================================================
    # TUMOR
    # =========================================================================

    if tumor_agent is None:

        path = DEFAULT_MODEL_PATHS["tumor"]

        try:

            print("\nLoading Tumor Classification Agent...")

            if not os.path.exists(path):
                raise FileNotFoundError(path)

            tumor_agent = TumorClassificationAgent(
                path
            )

            print("✓ Tumor Classification Agent loaded")

        except Exception as e:

            print(f"✗ Tumor Classification Agent failed: {e}")

    # =========================================================================
    # SEGMENTATION
    # =========================================================================

    if segmentation_agent is None:

        path = DEFAULT_MODEL_PATHS["segmentation"]

        try:

            print("\nLoading Liver Segmentation Agent...")

            if not os.path.exists(path):
                raise FileNotFoundError(path)

            segmentation_agent = LiverSegmentationAgent(
                model_path=path
            )

            print("✓ Liver Segmentation Agent loaded")

        except Exception as e:

            print(f"✗ Liver Segmentation Agent failed: {e}")

    # =========================================================================
    # CLINICAL REASONING
    # =========================================================================

    if clinical_agent is None:

        path = DEFAULT_MODEL_PATHS["clinical"]

        try:

            print("\nLoading Clinical Reasoning Agent...")

            if not os.path.exists(path):
                raise FileNotFoundError(path)

            clinical_agent = ClinicalReasoningAgent(
                path
            )

            print("✓ Clinical Reasoning Agent loaded")

        except Exception as e:

            print(f"✗ Clinical Reasoning Agent failed: {e}")

    # =========================================================================
    # SAVE AGENTS
    # =========================================================================

    self.cirrhosis_agent = cirrhosis_agent
    self.fatty_liver_agent = fatty_liver_agent
    self.fibrosis_agent = fibrosis_agent
    self.tumor_agent = tumor_agent
    self.segmentation_agent = segmentation_agent
    self.clinical_agent = clinical_agent

    # =========================================================================
    # REGISTRY
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
    # COORDINATION
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

    print("\n" + "=" * 80)
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
