# =============================================================================
# Liver AI Coordinator
# =============================================================================
#
# Main coordinator for the LiverAI-MultiAgent project.
#
# Available agents are loaded when their model is provided.
# Missing models do NOT prevent the coordinator from starting.
#
# =============================================================================

import os
import time


class LiverAICoordinator:

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        cirrhosis_model_path=None,
        fatty_liver_model_path=None,
        fibrosis_model_path=None,
        tumor_model_path=None,
        segmentation_model_path=None,
        clinical_reasoning_model_path=None
    ):

        print("\n" + "=" * 70)
        print("INITIALIZING LIVER AI COORDINATOR")
        print("=" * 70)

        # =====================================================================
        # AGENT CLASSES
        # =====================================================================

        self.CirrhosisAgent = None
        self.FattyLiverAgent = None
        self.FibrosisAgent = None
        self.TumorClassificationAgent = None
        self.LiverSegmentationAgent = None
        self.ClinicalReasoningAgent = None

        # =====================================================================
        # AGENT INSTANCES
        # =====================================================================

        self.cirrhosis_agent = None
        self.fatty_liver_agent = None
        self.fibrosis_agent = None
        self.tumor_agent = None
        self.segmentation_agent = None
        self.clinical_agent = None

        # =====================================================================
        # SAVE PATHS
        # =====================================================================

        self.cirrhosis_model_path = cirrhosis_model_path
        self.fatty_liver_model_path = fatty_liver_model_path
        self.fibrosis_model_path = fibrosis_model_path
        self.tumor_model_path = tumor_model_path
        self.segmentation_model_path = segmentation_model_path
        self.clinical_reasoning_model_path = (
            clinical_reasoning_model_path
        )

        # =====================================================================
        # IMPORT AGENTS
        # =====================================================================

        self._import_agents()

        # =====================================================================
        # LOAD AGENTS
        # =====================================================================

        self._load_cirrhosis()
        self._load_fatty_liver()
        self._load_fibrosis()
        self._load_tumor()
        self._load_segmentation()
        self._load_clinical_reasoning()

        print("\n" + "=" * 70)
        print("LIVER AI COORDINATOR READY")
        print("=" * 70)

    # =========================================================================
    # IMPORT AGENTS
    # =========================================================================

    def _import_agents(self):

        # ---------------------------------------------------------------------
        # Cirrhosis
        # ---------------------------------------------------------------------

        try:

            from agents.cirrhosis_agent import (
                CirrhosisAgent
            )

            self.CirrhosisAgent = CirrhosisAgent

        except Exception as e:

            print(
                "❌ Impossible d'importer CirrhosisAgent :",
                e
            )

        # ---------------------------------------------------------------------
        # Fatty Liver
        # ---------------------------------------------------------------------

        try:

            from agents.fatty_liver_agent import (
                FattyLiverAgent
            )

            self.FattyLiverAgent = FattyLiverAgent

        except Exception as e:

            print(
                "⚠️ Impossible d'importer FattyLiverAgent :",
                e
            )

        # ---------------------------------------------------------------------
        # Fibrosis
        # ---------------------------------------------------------------------

        try:

            from agents.fibrosis_agent import (
                FibrosisAgent
            )

            self.FibrosisAgent = FibrosisAgent

        except Exception as e:

            print(
                "⚠️ Impossible d'importer FibrosisAgent :",
                e
            )

        # ---------------------------------------------------------------------
        # Tumor
        # ---------------------------------------------------------------------

        try:

            from agents.tumor_classification_agent import (
                TumorClassificationAgent
            )

            self.TumorClassificationAgent = (
                TumorClassificationAgent
            )

        except Exception as e:

            print(
                "⚠️ Impossible d'importer TumorClassificationAgent :",
                e
            )

        # ---------------------------------------------------------------------
        # Segmentation
        # ---------------------------------------------------------------------

        try:

            from agents.liver_segmentation_agent import (
                LiverSegmentationAgent
            )

            self.LiverSegmentationAgent = (
                LiverSegmentationAgent
            )

        except Exception as e:

            print(
                "⚠️ Impossible d'importer LiverSegmentationAgent :",
                e
            )

        # ---------------------------------------------------------------------
        # Clinical Reasoning
        # ---------------------------------------------------------------------

        try:

            from agents.clinical_reasoning_agent import (
                ClinicalReasoningAgent
            )

            self.ClinicalReasoningAgent = (
                ClinicalReasoningAgent
            )

        except Exception as e:

            print(
                "⚠️ Impossible d'importer ClinicalReasoningAgent :",
                e
            )

    # =========================================================================
    # CIRRHOSIS
    # =========================================================================

    def _load_cirrhosis(self):

        if not self.cirrhosis_model_path:

            print(
                "⚠️ CirrhosisAgent : modèle non fourni"
            )

            return

        if not os.path.exists(
            self.cirrhosis_model_path
        ):

            print(
                "⚠️ Cirrhosis model absent :",
                self.cirrhosis_model_path
            )

            return

        if self.CirrhosisAgent is None:

            print(
                "⚠️ CirrhosisAgent indisponible"
            )

            return

        try:

            import joblib

            model_package = joblib.load(
                self.cirrhosis_model_path
            )

            self.cirrhosis_agent = (
                self.CirrhosisAgent(
                    model_package
                )
            )

            print(
                "✅ CirrhosisAgent chargé"
            )

        except Exception as e:

            print(
                "❌ Erreur CirrhosisAgent :",
                e
            )

            self.cirrhosis_agent = None

    # =========================================================================
    # FATTY LIVER
    # =========================================================================

    def _load_fatty_liver(self):

        if not self.fatty_liver_model_path:

            print(
                "⚠️ Fatty Liver Agent : modèle non fourni"
            )

            return

        if not os.path.exists(
            self.fatty_liver_model_path
        ):

            print(
                "⚠️ Fatty Liver model absent :",
                self.fatty_liver_model_path
            )

            return

        if self.FattyLiverAgent is None:

            print(
                "⚠️ FattyLiverAgent indisponible"
            )

            return

        try:

            import joblib

            model_package = joblib.load(
                self.fatty_liver_model_path
            )

            self.fatty_liver_agent = (
                self.FattyLiverAgent(
                    model_package
                )
            )

            print(
                "✅ FattyLiverAgent chargé"
            )

        except Exception as e:

            print(
                "❌ Erreur FattyLiverAgent :",
                e
            )

            self.fatty_liver_agent = None

    # =========================================================================
    # FIBROSIS
    # =========================================================================

    def _load_fibrosis(self):

        if not self.fibrosis_model_path:

            print(
                "⚠️ FibrosisAgent : modèle non fourni"
            )

            return

        if not os.path.exists(
            self.fibrosis_model_path
        ):

            print(
                "⚠️ Fibrosis model absent :",
                self.fibrosis_model_path
            )

            return

        if self.FibrosisAgent is None:

            print(
                "⚠️ FibrosisAgent indisponible"
            )

            return

        try:

            self.fibrosis_agent = (
                self.FibrosisAgent(
                    self.fibrosis_model_path
                )
            )

            print(
                "✅ FibrosisAgent chargé"
            )

        except Exception as e:

            print(
                "❌ Erreur FibrosisAgent :",
                e
            )

            self.fibrosis_agent = None

    # =========================================================================
    # TUMOR
    # =========================================================================

    def _load_tumor(self):

        if not self.tumor_model_path:

            print(
                "⚠️ TumorAgent : modèle non fourni"
            )

            return

        if not os.path.exists(
            self.tumor_model_path
        ):

            print(
                "⚠️ Tumor model absent :",
                self.tumor_model_path
            )

            return

        if self.TumorClassificationAgent is None:

            print(
                "⚠️ TumorClassificationAgent indisponible"
            )

            return

        try:

            self.tumor_agent = (
                self.TumorClassificationAgent(
                    self.tumor_model_path
                )
            )

            print(
                "✅ TumorClassificationAgent chargé"
            )

        except Exception as e:

            print(
                "❌ Erreur TumorClassificationAgent :",
                e
            )

            self.tumor_agent = None

    # =========================================================================
    # SEGMENTATION
    # =========================================================================

    def _load_segmentation(self):

        if not self.segmentation_model_path:

            print(
                "⚠️ SegmentationAgent : modèle non fourni"
            )

            return

        if not os.path.exists(
            self.segmentation_model_path
        ):

            print(
                "⚠️ Segmentation model absent :",
                self.segmentation_model_path
            )

            return

        if self.LiverSegmentationAgent is None:

            print(
                "⚠️ LiverSegmentationAgent indisponible"
            )

            return

        try:

            # LiverSegmentationAgent expects
            # the model path directly.

            self.segmentation_agent = (
                self.LiverSegmentationAgent(
                    self.segmentation_model_path
                )
            )

            print(
                "✅ LiverSegmentationAgent chargé"
            )

        except Exception as e:

            print(
                "❌ Erreur LiverSegmentationAgent :",
                e
            )

            self.segmentation_agent = None

    # =========================================================================
    # CLINICAL REASONING
    # =========================================================================

    def _load_clinical_reasoning(self):

        # ---------------------------------------------------------------------
        # No model supplied
        # ---------------------------------------------------------------------

        if not self.clinical_reasoning_model_path:

            print(
                "⚠️ ClinicalReasoningAgent : modèle non fourni"
            )

            return

        # ---------------------------------------------------------------------
        # Model path does not exist
        # ---------------------------------------------------------------------

        if not os.path.exists(
            self.clinical_reasoning_model_path
        ):

            print(
                "⚠️ Clinical reasoning model absent :",
                self.clinical_reasoning_model_path
            )

            return

        # ---------------------------------------------------------------------
        # Agent import failed
        # ---------------------------------------------------------------------

        if self.ClinicalReasoningAgent is None:

            print(
                "⚠️ ClinicalReasoningAgent indisponible"
            )

            return

        # ---------------------------------------------------------------------
        # Load model
        # ---------------------------------------------------------------------

        try:

            import joblib

            model_package = joblib.load(
                self.clinical_reasoning_model_path
            )

            self.clinical_agent = (
                self.ClinicalReasoningAgent(
                    model_package
                )
            )

            print(
                "✅ ClinicalReasoningAgent chargé"
            )

        except Exception as e:

            print(
                "❌ Erreur ClinicalReasoningAgent :",
                e
            )

            self.clinical_agent = None

    # =========================================================================
    # SINGLE PATIENT PREDICTION
    # =========================================================================

    def predict(self, patient_data):

        results = {}

        # ---------------------------------------------------------------------
        # Cirrhosis
        # ---------------------------------------------------------------------

        if self.cirrhosis_agent is not None:

            try:

                results["cirrhosis"] = (
                    self.cirrhosis_agent.predict(
                        patient_data
                    )
                )

            except Exception as e:

                results["cirrhosis"] = {
                    "agent": "CirrhosisAgent",
                    "status": "error",
                    "error": str(e)
                }

        # ---------------------------------------------------------------------
        # Fatty liver
        # ---------------------------------------------------------------------

        if self.fatty_liver_agent is not None:

            try:

                results["fatty_liver"] = (
                    self.fatty_liver_agent.predict(
                        patient_data
                    )
                )

            except Exception as e:

                results["fatty_liver"] = {
                    "agent": "FattyLiverAgent",
                    "status": "error",
                    "error": str(e)
                }

        # ---------------------------------------------------------------------
        # Fibrosis
        # ---------------------------------------------------------------------

        if self.fibrosis_agent is not None:

            try:

                results["fibrosis"] = (
                    self.fibrosis_agent.predict(
                        patient_data
                    )
                )

            except Exception as e:

                results["fibrosis"] = {
                    "agent": "FibrosisAgent",
                    "status": "error",
                    "error": str(e)
                }

        # ---------------------------------------------------------------------
        # Tumor
        # ---------------------------------------------------------------------

        if self.tumor_agent is not None:

            try:

                results["tumor"] = (
                    self.tumor_agent.predict(
                        patient_data
                    )
                )

            except Exception as e:

                results["tumor"] = {
                    "agent": "TumorClassificationAgent",
                    "status": "error",
                    "error": str(e)
                }

        # ---------------------------------------------------------------------
        # Segmentation
        # ---------------------------------------------------------------------

        if self.segmentation_agent is not None:

            try:

                results["segmentation"] = (
                    self.segmentation_agent.predict(
                        patient_data
                    )
                )

            except Exception as e:

                results["segmentation"] = {
                    "agent": "LiverSegmentationAgent",
                    "status": "error",
                    "error": str(e)
                }

        # ---------------------------------------------------------------------
        # Clinical reasoning
        # ---------------------------------------------------------------------

        if self.clinical_agent is not None:

            try:

                results["clinical_reasoning"] = (
                    self.clinical_agent.predict(
                        patient_data
                    )
                )

            except Exception as e:

                results["clinical_reasoning"] = {
                    "agent": "ClinicalReasoningAgent",
                    "status": "error",
                    "error": str(e)
                }

        return results

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_status(self):

        return {

            "cirrhosis": (
                self.cirrhosis_agent is not None
            ),

            "fatty_liver": (
                self.fatty_liver_agent is not None
            ),

            "fibrosis": (
                self.fibrosis_agent is not None
            ),

            "tumor": (
                self.tumor_agent is not None
            ),

            "segmentation": (
                self.segmentation_agent is not None
            ),

            "clinical_reasoning": (
                self.clinical_agent is not None
            )
        }

    # =========================================================================
    # SUMMARY
    # =========================================================================

    def summary(self):

        status = self.get_status()

        print("\n" + "=" * 70)
        print("LIVER AI COORDINATOR STATUS")
        print("=" * 70)

        for agent_name, available in status.items():

            symbol = "✅" if available else "⚠️"

            print(
                f"{symbol} {agent_name}: "
                f"{'AVAILABLE' if available else 'NOT AVAILABLE'}"
            )

        print("=" * 70)

        return status
