# ============================================================
# LiverAI Multi-Agent Orchestrator
# ============================================================

import traceback
from datetime import datetime


class LiverAIOrchestrator:

    def __init__(
        self,
        cirrhosis_agent=None,
        fatty_liver_agent=None,
        clinical_agent=None,
        fibrosis_agent=None,
        tumor_agent=None,
        segmentation_agent=None
    ):

        self.name = "LiverAI Orchestrator"

        self.cirrhosis_agent = cirrhosis_agent
        self.fatty_liver_agent = fatty_liver_agent
        self.clinical_agent = clinical_agent
        self.fibrosis_agent = fibrosis_agent
        self.tumor_agent = tumor_agent
        self.segmentation_agent = segmentation_agent

        self.agents = {

            "cirrhosis": cirrhosis_agent,

            "fatty_liver": fatty_liver_agent,

            "fibrosis": fibrosis_agent,

            "tumor_classification": tumor_agent,

            "liver_segmentation": segmentation_agent,

            "clinical_reasoning": clinical_agent
        }

        self.last_results = {}

        self.execution_log = []

        print("=" * 80)
        print("LIVERAI MULTI-AGENT ORCHESTRATOR")
        print("=" * 80)

        for name, agent in self.agents.items():

            if agent is not None:

                print(
                    f"✅ {name:<25} "
                    f"{agent.__class__.__name__}"
                )

            else:

                print(
                    f"⚠️ {name:<25} NOT AVAILABLE"
                )

        print("=" * 80)

    # =========================================================
    # LOG
    # =========================================================

    def _log(self, message):

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        self.execution_log.append({

            "timestamp": timestamp,

            "message": message
        })

        print(message)

    # =========================================================
    # SAFE AGENT EXECUTION
    # =========================================================

    def _execute_agent(
        self,
        agent_name,
        agent,
        input_data
    ):

        if agent is None:

            return {

                "agent": agent_name,

                "status": "not_available",

                "prediction": None,

                "probability": None,

                "confidence": 0.0,

                "uncertainty": 1.0,

                "quality": 0.0,

                "details": {},

                "error": "Agent not available"
            }

        if input_data is None:

            return {

                "agent": agent_name,

                "status": "no_input",

                "prediction": None,

                "probability": None,

                "confidence": 0.0,

                "uncertainty": 1.0,

                "quality": 0.0,

                "details": {},

                "error": "No input provided"
            }

        self._log(
            f"\n[{agent_name}] START"
        )

        try:

            result = agent.predict(
                input_data
            )

            if result is None:

                result = {}

            if not isinstance(
                result,
                dict
            ):

                result = {

                    "prediction": result
                }

            result.setdefault(
                "agent",
                agent_name
            )

            result.setdefault(
                "status",
                "completed"
            )

            result.setdefault(
                "probability",
                None
            )

            result.setdefault(
                "confidence",
                result.get(
                    "probability",
                    0.0
                )
                if result.get(
                    "probability"
                ) is not None
                else 0.0
            )

            result.setdefault(
                "uncertainty",
                1.0 - result.get(
                    "confidence",
                    0.0
                )
            )

            result.setdefault(
                "quality",
                1.0
            )

            result.setdefault(
                "details",
                {}
            )

            result.setdefault(
                "error",
                None
            )

            self._log(
                f"[{agent_name}] ✅ COMPLETED"
            )

            return result

        except Exception as e:

            self._log(
                f"[{agent_name}] ❌ ERROR: {e}"
            )

            traceback.print_exc()

            return {

                "agent": agent_name,

                "status": "error",

                "prediction": None,

                "probability": None,

                "confidence": 0.0,

                "uncertainty": 1.0,

                "quality": 0.0,

                "details": {},

                "error": str(e)
            }

    # =========================================================
    # RUN SPECIALIZED AGENTS
    # =========================================================

    def run_specialized_agents(
        self,
        clinical_data=None,
        image=None,
        volume=None
    ):

        results = {}

        # -----------------------------------------------------
        # CIRRHOSIS
        # -----------------------------------------------------

        if clinical_data is not None:

            results["cirrhosis"] = (
                self._execute_agent(

                    "CirrhosisAgent",

                    self.cirrhosis_agent,

                    clinical_data
                )
            )

        # -----------------------------------------------------
        # FATTY LIVER
        # -----------------------------------------------------

        if clinical_data is not None:

            results["fatty_liver"] = (
                self._execute_agent(

                    "FattyLiverAgent",

                    self.fatty_liver_agent,

                    clinical_data
                )
            )

        # -----------------------------------------------------
        # FIBROSIS
        # -----------------------------------------------------

        if clinical_data is not None:

            results["fibrosis"] = (
                self._execute_agent(

                    "FibrosisAgent",

                    self.fibrosis_agent,

                    clinical_data
                )
            )

        # -----------------------------------------------------
        # TUMOR
        # -----------------------------------------------------

        if image is not None:

            results["tumor_classification"] = (
                self._execute_agent(

                    "TumorClassificationAgent",

                    self.tumor_agent,

                    image
                )
            )

        # -----------------------------------------------------
        # SEGMENTATION
        # -----------------------------------------------------

        if volume is not None:

            results["liver_segmentation"] = (
                self._execute_agent(

                    "LiverSegmentationAgent",

                    self.segmentation_agent,

                    volume
                )
            )

        return results

    # =========================================================
    # CLINICAL REASONING
    # =========================================================

    def run_clinical_reasoning(
        self,
        agent_results
    ):

        if self.clinical_agent is None:

            return {

                "agent":
                    "ClinicalReasoningAgent",

                "status":
                    "not_available",

                "error":
                    "Clinical reasoning agent not available"
            }

        self._log(
            "\n[ClinicalReasoningAgent] START"
        )

        try:

            # IMPORTANT:
            # ClinicalReasoningAgent receives
            # the results of the other agents,
            # NOT the original clinical_data.

            result = self.clinical_agent.predict(
                agent_results
            )

            if result is None:

                result = {}

            if not isinstance(
                result,
                dict
            ):

                result = {

                    "prediction": result
                }

            result.setdefault(
                "agent",
                "ClinicalReasoningAgent"
            )

            result.setdefault(
                "status",
                "completed"
            )

            self._log(
                "[ClinicalReasoningAgent] ✅ COMPLETED"
            )

            return result

        except Exception as e:

            self._log(
                f"[ClinicalReasoningAgent] ❌ ERROR: {e}"
            )

            traceback.print_exc()

            return {

                "agent":
                    "ClinicalReasoningAgent",

                "status":
                    "error",

                "error":
                    str(e)
            }

    # =========================================================
    # COMPLETE PIPELINE
    # =========================================================

    def run(
        self,
        patient_id,
        clinical_data=None,
        image=None,
        volume=None
    ):

        print("\n")
        print("=" * 80)
        print(
            f"LIVERAI PIPELINE — PATIENT {patient_id}"
        )
        print("=" * 80)

        # -----------------------------------------------------
        # STEP 1
        # Specialized agents
        # -----------------------------------------------------

        agent_results = (
            self.run_specialized_agents(

                clinical_data=clinical_data,

                image=image,

                volume=volume
            )
        )

        # -----------------------------------------------------
        # STEP 2
        # Clinical reasoning
        # -----------------------------------------------------

        clinical_result = (
            self.run_clinical_reasoning(
                agent_results
            )
        )

        # -----------------------------------------------------
        # STEP 3
        # Add clinical reasoning
        # -----------------------------------------------------

        agent_results[
            "clinical_reasoning"
        ] = clinical_result

        # -----------------------------------------------------
        # SAVE
        # -----------------------------------------------------

        self.last_results = agent_results

        # -----------------------------------------------------
        # FINAL RESULT
        # -----------------------------------------------------

        final_result = {

            "patient_id":
                patient_id,

            "status":
                "completed",

            "agents":
                agent_results,

            "clinical_reasoning":
                clinical_result
        }

        print("\n")
        print("=" * 80)
        print("LIVERAI PIPELINE COMPLETED")
        print("=" * 80)

        return final_result
