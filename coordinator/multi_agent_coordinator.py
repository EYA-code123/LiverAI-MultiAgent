import os
import joblib

from agents.cirrhosis_agent import CirrhosisAgent
from agents.fatty_liver_agent import FattyLiverAgent
from agents.fibrosis_agent import FibrosisAgent
from agents.tumor_classification_agent import TumorClassificationAgent
from agents.liver_segmentation_agent import LiverSegmentationAgent
from agents.clinical_reasoning_agent import ClinicalReasoningAgent


class MultiAgentCoordinator:

    def __init__(self, model_paths):

        self.name = "Liver Multi-Agent Coordinator"

        self.model_paths = model_paths

        self.agents = {}

        self._initialize_agents()

        self.clinical_agent = ClinicalReasoningAgent()

    # ==========================================================
    # INITIALIZE AGENTS
    # ==========================================================

    def _initialize_agents(self):

        # ------------------------------------------------------
        # CIRRHOSIS
        # ------------------------------------------------------

        path = self.model_paths.get("cirrhosis")

        if path and os.path.exists(path):

            try:

                package = joblib.load(path)

                self.agents["cirrhosis"] = \
                    CirrhosisAgent(package)

                print("✅ CirrhosisAgent")

            except Exception as e:

                print(
                    "❌ CirrhosisAgent:",
                    e
                )

        else:

            print("⚠️ Cirrhosis model unavailable")

        # ------------------------------------------------------
        # FATTY LIVER
        # ------------------------------------------------------

        path = self.model_paths.get("fatty_liver")

        if path and os.path.exists(path):

            try:

                package = joblib.load(path)

                self.agents["fatty_liver"] = \
                    FattyLiverAgent(package)

                print("✅ FattyLiverAgent")

            except Exception as e:

                print(
                    "❌ FattyLiverAgent:",
                    e
                )

        else:

            print("⚠️ Fatty Liver model unavailable")

        # ------------------------------------------------------
        # FIBROSIS
        # ------------------------------------------------------

        path = self.model_paths.get("fibrosis")

        if path and os.path.exists(path):

            try:

                model = joblib.load(path)

                self.agents["fibrosis"] = \
                    FibrosisAgent(model)

                print("✅ FibrosisAgent")

            except Exception as e:

                print(
                    "❌ FibrosisAgent:",
                    e
                )

        else:

            print("⚠️ Fibrosis model unavailable")

        # ------------------------------------------------------
        # TUMOR
        # ------------------------------------------------------

        path = self.model_paths.get("tumor")

        if path and os.path.exists(path):

            try:

                self.agents["tumor_classification"] = \
                    TumorClassificationAgent(path)

                print("✅ TumorClassificationAgent")

            except Exception as e:

                print(
                    "❌ TumorClassificationAgent:",
                    e
                )

        else:

            print("⚠️ Tumor model unavailable")

        # ------------------------------------------------------
        # SEGMENTATION
        # ------------------------------------------------------

        path = self.model_paths.get("segmentation")

        if path and os.path.exists(path):

            try:

                self.agents["liver_segmentation"] = \
                    LiverSegmentationAgent(path)

                print("✅ LiverSegmentationAgent")

            except Exception as e:

                print(
                    "❌ LiverSegmentationAgent:",
                    e
                )

        else:

            print("⚠️ Segmentation model unavailable")

    # ==========================================================
    # RUN COORDINATION
    # ==========================================================

    def predict(
        self,
        patient_data=None,
        image=None
    ):

        results = {}

        # ======================================================
        # CIRRHOSIS
        # ======================================================

        if "cirrhosis" in self.agents:

            try:

                results["cirrhosis"] = \
                    self.agents["cirrhosis"].predict(
                        patient_data
                    )

            except Exception as e:

                results["cirrhosis"] = {

                    "agent":
                        "CirrhosisAgent",

                    "status":
                        "error",

                    "prediction":
                        None,

                    "probability":
                        None,

                    "error":
                        str(e)
                }

        else:

            results["cirrhosis"] = {

                "status":
                    "not_available",

                "prediction":
                    None,

                "probability":
                    None
            }

        # ======================================================
        # FATTY LIVER
        # ======================================================

        if "fatty_liver" in self.agents:

            try:

                results["fatty_liver"] = \
                    self.agents["fatty_liver"].predict(
                        patient_data
                    )

            except Exception as e:

                results["fatty_liver"] = {

                    "agent":
                        "FattyLiverAgent",

                    "status":
                        "error",

                    "prediction":
                        None,

                    "probability":
                        None,

                    "error":
                        str(e)
                }

        else:

            results["fatty_liver"] = {

                "status":
                    "not_available",

                "prediction":
                    None,

                "probability":
                    None
            }

        # ======================================================
        # FIBROSIS
        # ======================================================

        if "fibrosis" in self.agents:

            try:

                results["fibrosis"] = \
                    self.agents["fibrosis"].predict(
                        patient_data
                    )

            except Exception as e:

                results["fibrosis"] = {

                    "agent":
                        "FibrosisAgent",

                    "status":
                        "error",

                    "prediction":
                        None,

                    "probability":
                        None,

                    "error":
                        str(e)
                }

        else:

            results["fibrosis"] = {

                "status":
                    "not_available",

                "prediction":
                    None,

                "probability":
                    None
            }

        # ======================================================
        # TUMOR
        # ======================================================

        if "tumor_classification" in self.agents:

            if image is not None:

                try:

                    results[
                        "tumor_classification"
                    ] = self.agents[
                        "tumor_classification"
                    ].predict(image)

                except Exception as e:

                    results[
                        "tumor_classification"
                    ] = {

                        "agent":
                            "TumorClassificationAgent",

                        "status":
                            "error",

                        "prediction":
                            None,

                        "probability":
                            None,

                        "error":
                            str(e)
                    }

            else:

                results[
                    "tumor_classification"
                ] = {

                    "status":
                        "not_available",

                    "prediction":
                        None,

                    "probability":
                        None,

                    "reason":
                        "No medical image provided"
                }

        else:

            results[
                "tumor_classification"
            ] = {

                "status":
                    "not_available",

                "prediction":
                    None,

                "probability":
                    None
            }

        # ======================================================
        # SEGMENTATION
        # ======================================================

        if "liver_segmentation" in self.agents:

            if image is not None:

                try:

                    results[
                        "liver_segmentation"
                    ] = self.agents[
                        "liver_segmentation"
                    ].predict(image)

                except Exception as e:

                    results[
                        "liver_segmentation"
                    ] = {

                        "agent":
                            "LiverSegmentationAgent",

                        "status":
                            "error",

                        "prediction":
                            None,

                        "error":
                            str(e)
                    }

            else:

                results[
                    "liver_segmentation"
                ] = {

                    "status":
                        "not_available",

                    "prediction":
                        None,

                    "reason":
                        "No medical image provided"
                }

        else:

            results[
                "liver_segmentation"
            ] = {

                "status":
                    "not_available",

                "prediction":
                    None
            }

        # ======================================================
        # CLINICAL REASONING
        # ======================================================

        clinical_result = \
            self.clinical_agent.predict(
                results
            )

        # ======================================================
        # FINAL OUTPUT
        # ======================================================

        return {

            "coordinator":
                self.name,

            "status":
                "completed",

            "agent_results":
                results,

            "clinical_reasoning":
                clinical_result
        }
