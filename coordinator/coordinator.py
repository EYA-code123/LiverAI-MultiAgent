# ============================================================
# LIVER AI MULTI-AGENT COORDINATOR
# ============================================================

import os
import joblib
import numpy as np


class LiverAICoordinator:

    def __init__(
        self,
        cirrhosis_model_path,
        fatty_liver_model_path,
        fibrosis_model_path=None,
        tumor_model_path=None,
        segmentation_model_path=None
    ):

        self.name = "LiverAICoordinator"

        # ====================================================
        # PATHS
        # ====================================================

        self.cirrhosis_model_path = cirrhosis_model_path
        self.fatty_liver_model_path = fatty_liver_model_path
        self.fibrosis_model_path = fibrosis_model_path
        self.tumor_model_path = tumor_model_path
        self.segmentation_model_path = segmentation_model_path

        # ====================================================
        # IMPORT AGENTS
        # ====================================================

        from agents.cirrhosis_agent import CirrhosisAgent
        from agents.fatty_liver_agent import FattyLiverAgent
        from agents.fibrosis_agent import FibrosisAgent
        from agents.tumor_classification_agent import TumorClassificationAgent
        from agents.liver_segmentation_agent import LiverSegmentationAgent
        from agents.clinical_reasoning_agent import ClinicalReasoningAgent

        self.CirrhosisAgent = CirrhosisAgent
        self.FattyLiverAgent = FattyLiverAgent
        self.FibrosisAgent = FibrosisAgent
        self.TumorClassificationAgent = TumorClassificationAgent
        self.LiverSegmentationAgent = LiverSegmentationAgent
        self.ClinicalReasoningAgent = ClinicalReasoningAgent

        # ====================================================
        # LOAD AVAILABLE MODELS
        # ====================================================

        self.cirrhosis_agent = None
        self.fatty_liver_agent = None
        self.fibrosis_agent = None
        self.tumor_agent = None
        self.segmentation_agent = None

        self._load_cirrhosis()
        self._load_fatty_liver()
        self._load_fibrosis()
        self._load_tumor()
        self._load_segmentation()

        # ====================================================
        # CLINICAL REASONING
        # ====================================================

        self.clinical_agent = self.ClinicalReasoningAgent()

    # ========================================================
    # CIRRHOSIS
    # ========================================================

    def _load_cirrhosis(self):

        if not os.path.exists(self.cirrhosis_model_path):

            print(
                "⚠️ Cirrhosis model absent :",
                self.cirrhosis_model_path
            )

            return

        try:

            package = joblib.load(
                self.cirrhosis_model_path
            )

            self.cirrhosis_agent = self.CirrhosisAgent(
                package
            )

            print(
                "✅ CirrhosisAgent chargé"
            )

        except Exception as e:

            print(
                "❌ Erreur CirrhosisAgent :",
                e
            )

    # ========================================================
    # FATTY LIVER
    # ========================================================

    def _load_fatty_liver(self):

        if not os.path.exists(
            self.fatty_liver_model_path
        ):

            print(
                "⚠️ Fatty Liver model absent :",
                self.fatty_liver_model_path
            )

            return

        try:

            package = joblib.load(
                self.fatty_liver_model_path
            )

            self.fatty_liver_agent = (
                self.FattyLiverAgent(
                    package
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

    # ========================================================
    # FIBROSIS
    # ========================================================

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

        try:

            model = joblib.load(
                self.fibrosis_model_path
            )

            self.fibrosis_agent = (
                self.FibrosisAgent(
                    model
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

    # ========================================================
    # TUMOR
    # ========================================================

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

    # ========================================================
    # SEGMENTATION
    # ========================================================

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

        try:

            model = joblib.load(
                self.segmentation_model_path
            )

            self.segmentation_agent = (
                self.LiverSegmentationAgent(
                    model
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

    # ========================================================
    # SAFE RESULT
    # ========================================================

    def _not_available(self, agent_name):

        return {

            "agent": agent_name,

            "prediction": None,

            "probability": None,

            "status": "not_available",

            "error": None
        }

    # ========================================================
    # RUN ALL AGENTS
    # ========================================================

    def predict(
        self,
        patient_data,
        image=None
    ):

        results = {}

        # ====================================================
        # CIRRHOSIS
        # ====================================================

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

                    "prediction": None,

                    "probability": None,

                    "status": "error",

                    "error": str(e)
                }

        else:

            results["cirrhosis"] = (
                self._not_available(
                    "CirrhosisAgent"
                )
            )

        # ====================================================
        # FATTY LIVER
        # ====================================================

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

                    "prediction": None,

                    "probability": None,

                    "status": "error",

                    "error": str(e)
                }

        else:

            results["fatty_liver"] = (
                self._not_available(
                    "FattyLiverAgent"
                )
            )

        # ====================================================
        # FIBROSIS
        # ====================================================

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

                    "prediction": None,

                    "probability": None,

                    "status": "error",

                    "error": str(e)
                }

        else:

            results["fibrosis"] = (
                self._not_available(
                    "FibrosisAgent"
                )
            )

        # ====================================================
        # TUMOR
        # ====================================================

        if self.tumor_agent is not None and image is not None:

            try:

                results["tumor_classification"] = (
                    self.tumor_agent.predict(
                        image
                    )
                )

            except Exception as e:

                results["tumor_classification"] = {

                    "agent":
                        "TumorClassificationAgent",

                    "prediction": None,

                    "probability": None,

                    "status": "error",

                    "error": str(e)
                }

        else:

            results["tumor_classification"] = (
                self._not_available(
                    "TumorClassificationAgent"
                )
            )

        # ====================================================
        # SEGMENTATION
        # ====================================================

        if self.segmentation_agent is not None and image is not None:

            try:

                results["liver_segmentation"] = (
                    self.segmentation_agent.predict(
                        image
                    )
                )

            except Exception as e:

                results["liver_segmentation"] = {

                    "agent":
                        "LiverSegmentationAgent",

                    "prediction": None,

                    "probability": None,

                    "status": "error",

                    "error": str(e)
                }

        else:

            results["liver_segmentation"] = (
                self._not_available(
                    "LiverSegmentationAgent"
                )
            )

        # ====================================================
        # CLINICAL REASONING
        # ====================================================

        try:

            clinical_result = (
                self.clinical_agent.predict(
                    results
                )
            )

        except Exception as e:

            clinical_result = {

                "agent":
                    "Clinical Reasoning Agent",

                "status": "error",

                "error": str(e)
            }

        # ====================================================
        # FINAL OUTPUT
        # ====================================================

        return {

            "agents": results,

            "clinical_reasoning":
                clinical_result
        }
