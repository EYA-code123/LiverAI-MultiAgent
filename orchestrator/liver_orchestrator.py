```python
# ============================================================
# LiverAI Orchestrator
# orchestrator/liver_orchestrator.py
# ============================================================

from datetime import datetime
import traceback


class LiverAIOrchestrator:

    def __init__(
        self,
        fatty_agent=None,
        fibrosis_agent=None,
        cirrhosis_agent=None,
        tumor_agent=None,
        segmentation_agent=None,
        clinical_reasoning_agent=None
    ):

        self.name = "LiverAI Orchestrator"

        self.fatty_agent = fatty_agent
        self.fibrosis_agent = fibrosis_agent
        self.cirrhosis_agent = cirrhosis_agent
        self.tumor_agent = tumor_agent
        self.segmentation_agent = segmentation_agent
        self.clinical_reasoning_agent = (
            clinical_reasoning_agent
        )

        self.agents = {
            "fatty_liver": fatty_agent,
            "fibrosis": fibrosis_agent,
            "cirrhosis": cirrhosis_agent,
            "tumor_classification": tumor_agent,
            "liver_segmentation": segmentation_agent,
            "clinical_reasoning": clinical_reasoning_agent
        }

        self.last_results = {}
        self.last_assessment = None
        self.execution_log = []

        print("=" * 80)
        print("LIVERAI ORCHESTRATOR")
        print("=" * 80)

        for name, agent in self.agents.items():

            if agent is not None:
                print(
                    f"✓ {name}: "
                    f"{agent.__class__.__name__}"
                )
            else:
                print(
                    f"⚠ {name}: unavailable"
                )

        print("=" * 80)

    # ========================================================
    # LOG
    # ========================================================

    def _log(self, message):

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        self.execution_log.append({
            "timestamp": timestamp,
            "message": message
        })

        print(message)

    # ========================================================
    # SAFE EXECUTION
    # ========================================================

    def _execute_agent(
        self,
        name,
        agent,
        input_data
    ):

        if agent is None:

            return {
                "agent": name,
                "status": "not_available",
                "prediction": None,
                "probability": None
            }

        if input_data is None:

            return {
                "agent": name,
                "status": "no_input",
                "prediction": None,
                "probability": None
            }

        self._log(
            f"[{name}] START"
        )

        try:

            if hasattr(agent, "predict"):

                result = agent.predict(
                    input_data
                )

            elif hasattr(agent, "analyze"):

                result = agent.analyze(
                    input_data
                )

            else:

                raise AttributeError(
                    f"{name} has no predict/analyze method"
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

            result["agent"] = name

            result.setdefault(
                "status",
                "completed"
            )

            self._log(
                f"[{name}] ✓ COMPLETED"
            )

            return result

        except Exception as e:

            self._log(
                f"[{name}] ✗ ERROR: {e}"
            )

            traceback.print_exc()

            return {
                "agent": name,
                "status": "error",
                "prediction": None,
                "probability": None,
                "error": str(e)
            }

    # ========================================================
    # MAIN PIPELINE
    # ========================================================

    def analyze(
        self,
        clinical_data=None,
        tumor_image=None,
        liver_volume=None
    ):

        self.execution_log = []

        print()
        print("=" * 80)
        print("LIVERAI MULTI-AGENT ANALYSIS")
        print("=" * 80)

        results = {}

        # ====================================================
        # PHASE 1
        # SPECIALIZED AGENTS
        # ====================================================

        print()
        print(
            "PHASE 1 — SPECIALIZED ANALYSIS"
        )

        # ----------------------------------------------------
        # FAT
        # ----------------------------------------------------

        results["fatty_liver"] = (
            self._execute_agent(
                "fatty_liver",
                self.fatty_agent,
                clinical_data
            )
        )

        # ----------------------------------------------------
        # FIBROSIS
        # ----------------------------------------------------

        results["fibrosis"] = (
            self._execute_agent(
                "fibrosis",
                self.fibrosis_agent,
                clinical_data
            )
        )

        # ----------------------------------------------------
        # CIRRHOSIS
        # ----------------------------------------------------

        results["cirrhosis"] = (
            self._execute_agent(
                "cirrhosis",
                self.cirrhosis_agent,
                clinical_data
            )
        )

        # ----------------------------------------------------
        # TUMOR
        # ----------------------------------------------------

        results["tumor_classification"] = (
            self._execute_agent(
                "tumor_classification",
                self.tumor_agent,
                tumor_image
            )
        )

        # ----------------------------------------------------
        # SEGMENTATION
        # ----------------------------------------------------

        results["liver_segmentation"] = (
            self._execute_agent(
                "liver_segmentation",
                self.segmentation_agent,
                liver_volume
            )
        )

        # ====================================================
        # PHASE 2
        # CLINICAL REASONING
        # ====================================================

        print()
        print(
            "PHASE 2 — CLINICAL REASONING"
        )

        clinical_result = (
            self._execute_agent(
                "clinical_reasoning",
                self.clinical_reasoning_agent,
                results
            )
        )

        results[
            "clinical_reasoning"
        ] = clinical_result

        # ====================================================
        # PHASE 3
        # UNIFIED ASSESSMENT
        # ====================================================

        print()
        print(
            "PHASE 3 — UNIFIED ASSESSMENT"
        )

        unified = (
            self._build_unified_assessment(
                results
            )
        )

        results[
            "unified_assessment"
        ] = unified

        self.last_results = results
        self.last_assessment = unified

        print()
        print("=" * 80)
        print(
            "LIVERAI ANALYSIS COMPLETED"
        )
        print("=" * 80)

        return results

    # ========================================================
    # UNIFIED ASSESSMENT
    # ========================================================

    def _build_unified_assessment(
        self,
        results
    ):

        clinical = results.get(
            "clinical_reasoning",
            {}
        )

        completed_agents = []

        for name, result in results.items():

            if not isinstance(
                result,
                dict
            ):
                continue

            if result.get(
                "status"
            ) in [
                "completed",
                "success"
            ]:

                completed_agents.append(
                    name
                )

        return {

            "system":
                "LiverAI Multi-Agent System",

            "completed_agents":
                completed_agents,

            "number_of_completed_agents":
                len(completed_agents),

            "overall_risk":
                clinical.get(
                    "overall_risk"
                ),

            "findings":
                clinical.get(
                    "findings",
                    []
                ),

            "tumor_detected":
                clinical.get(
                    "tumor_detected",
                    False
                ),

            "fatty_liver":
                clinical.get(
                    "fatty_liver_prediction"
                ),

            "fibrosis":
                clinical.get(
                    "fibrosis_prediction"
                ),

            "cirrhosis":
                clinical.get(
                    "cirrhosis_prediction"
                ),

            "tumor":
                clinical.get(
                    "tumor_prediction"
                ),

            "segmentation_available":
                clinical.get(
                    "segmentation_available",
                    False
                ),

            "status":
                "completed"
        }

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(self):

        return {

            name: {

                "available":
                    agent is not None,

                "class":
                    (
                        agent.__class__.__name__
                        if agent is not None
                        else None
                    )

            }

            for name, agent
            in self.agents.items()
        }
```
