```python
# ================================================================
# LiverAI-MultiAgent
# COMPLETE MULTI-AGENT ORCHESTRATOR
# ================================================================
#
# Agents:
#   1. Fatty Liver
#   2. Fibrosis
#   3. Cirrhosis
#   4. Tumor Classification
#   5. Liver Segmentation
#   6. Clinical Reasoning
#
# Pipeline:
#
# Patient Data
#      |
#      v
# LiverAI Orchestrator
#      |
#      +--> Fatty Liver
#      +--> Fibrosis
#      +--> Cirrhosis
#      +--> Tumor
#      +--> Segmentation
#      |
#      v
# Adaptive Fusion
#      |
#      v
# Conflict Detection
#      |
#      v
# Clinical Reasoning
#      |
#      v
# Decision Engine
#      |
#      v
# Unified Liver Assessment
#
# ================================================================

import time
import traceback
from datetime import datetime


# ================================================================
# OPTIONAL COORDINATORS
# ================================================================

try:
    from coordinator.trust import TrustManager
except Exception:
    TrustManager = None

try:
    from coordinator.adaptive_fusion import AdaptiveFusion
except Exception:
    AdaptiveFusion = None

try:
    from coordinator.conflict import ConflictDetector
except Exception:
    ConflictDetector = None

try:
    from coordinator.decision import DecisionEngine
except Exception:
    DecisionEngine = None

try:
    from orchestrator.schemas import AgentResult
except Exception:
    AgentResult = None


# ================================================================
# MAIN ORCHESTRATOR
# ================================================================

class LiverAIOrchestrator:

    # ------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------

    def __init__(
        self,
        fatty_agent=None,
        fibrosis_agent=None,
        cirrhosis_agent=None,
        tumor_agent=None,
        segmentation_agent=None,
        clinical_reasoning_agent=None,
        clinical_agent=None,
        fatty_liver_agent=None,
        **kwargs
    ):

        # --------------------------------------------------------
        # Backward compatibility
        # --------------------------------------------------------

        if fatty_agent is None:
            fatty_agent = fatty_liver_agent

        if clinical_reasoning_agent is None:
            clinical_reasoning_agent = clinical_agent

        self.name = "LiverAI Orchestrator"

        # --------------------------------------------------------
        # Agents
        # --------------------------------------------------------

        self.fatty_agent = fatty_agent
        self.fibrosis_agent = fibrosis_agent
        self.cirrhosis_agent = cirrhosis_agent
        self.tumor_agent = tumor_agent
        self.segmentation_agent = segmentation_agent
        self.clinical_reasoning_agent = clinical_reasoning_agent

        # --------------------------------------------------------
        # Registry
        # --------------------------------------------------------

        self.agents = {
            "fatty_liver": self.fatty_agent,
            "fibrosis": self.fibrosis_agent,
            "cirrhosis": self.cirrhosis_agent,
            "tumor_classification": self.tumor_agent,
            "liver_segmentation": self.segmentation_agent,
            "clinical_reasoning": self.clinical_reasoning_agent,
        }

        # --------------------------------------------------------
        # Coordinators
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # Runtime state
        # --------------------------------------------------------

        self.last_results = {}
        self.last_assessment = None
        self.execution_log = []

        print("=" * 80)
        print("LIVERAI MULTI-AGENT SYSTEM")
        print("=" * 80)

        print("\nRegistered Agents:")

        for name, agent in self.agents.items():

            status = "✓" if agent is not None else "✗"

            print(
                f"  {status} {name}"
            )

        print("\n✓ LiverAI Orchestrator initialized")
        print("=" * 80)

    # ============================================================
    # LOGGING
    # ============================================================

    def _log(self, message):

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        self.execution_log.append(
            {
                "timestamp": timestamp,
                "message": str(message),
            }
        )

        print(message)

    # ============================================================
    # UTILITY
    # ============================================================

    @staticmethod
    def _safe_float(value, default=0.0):

        try:
            return float(value)

        except Exception:
            return default

    @staticmethod
    def _safe_probability(value):

        try:

            value = float(value)

            if value < 0:
                return 0.0

            if value > 1:
                return 1.0

            return value

        except Exception:

            return 0.0

    # ============================================================
    # AGENT EXECUTION
    # ============================================================

    def _execute_agent(
        self,
        agent_name,
        agent,
        input_data
    ):

        self._log(
            f"\n[{agent_name.upper()}] Starting..."
        )

        # --------------------------------------------------------
        # Agent unavailable
        # --------------------------------------------------------

        if agent is None:

            self._log(
                f"[{agent_name.upper()}] "
                "Agent unavailable."
            )

            return {
                "agent_id": agent_name,
                "agent": agent_name,
                "task_type": agent_name,
                "status": "not_available",
                "prediction": None,
                "probability": 0.0,
                "confidence": 0.0,
                "uncertainty": 1.0,
                "quality": 0.0,
                "trust": 0.0,
                "latency_ms": 0.0,
                "missing_data_ratio": 1.0,
            }

        # --------------------------------------------------------
        # No input
        # --------------------------------------------------------

        if input_data is None:

            self._log(
                f"[{agent_name.upper()}] "
                "No input → skipped."
            )

            return {
                "agent_id": agent_name,
                "agent": agent_name,
                "task_type": agent_name,
                "status": "not_run",
                "prediction": None,
                "probability": 0.0,
                "confidence": 0.0,
                "uncertainty": 1.0,
                "quality": 0.0,
                "trust": 0.0,
                "latency_ms": 0.0,
                "missing_data_ratio": 1.0,
            }

        # --------------------------------------------------------
        # Execute
        # --------------------------------------------------------

        start = time.perf_counter()

        try:

            # Most of our agents use predict()
            if hasattr(agent, "predict"):

                try:
                    result = agent.predict(input_data)

                except TypeError:

                    result = agent.predict(
                        input_data=input_data
                    )

            # Some agents may expose analyze()
            elif hasattr(agent, "analyze"):

                try:
                    result = agent.analyze(input_data)

                except TypeError:

                    result = agent.analyze(
                        input_data=input_data
                    )

            # Some implementations use run()
            elif hasattr(agent, "run"):

                try:
                    result = agent.run(input_data)

                except TypeError:

                    result = agent.run(
                        input_data=input_data
                    )

            else:

                raise AttributeError(
                    f"{agent_name} has no "
                    "predict(), analyze() or run()."
                )

            latency_ms = (
                time.perf_counter() - start
            ) * 1000.0

            # ----------------------------------------------------
            # Normalize
            # ----------------------------------------------------

            if result is None:

                result = {}

            if not isinstance(result, dict):

                result = {
                    "prediction": result
                }

            else:

                # Copy to avoid modifying agent-owned dictionary
                result = dict(result)

            result["agent_id"] = agent_name
            result["agent"] = agent_name

            result.setdefault(
                "task_type",
                agent_name
            )

            result.setdefault(
                "status",
                "success"
            )

            result.setdefault(
                "latency_ms",
                latency_ms
            )

            # ----------------------------------------------------
            # Confidence
            # ----------------------------------------------------

            confidence = result.get(
                "confidence",
                result.get(
                    "probability",
                    0.0
                )
            )

            confidence = self._safe_probability(
                confidence
            )

            result["confidence"] = confidence

            # ----------------------------------------------------
            # Probability
            # ----------------------------------------------------

            probability = result.get(
                "probability",
                confidence
            )

            if isinstance(
                probability,
                (list, tuple)
            ):

                result["class_probabilities"] = (
                    list(probability)
                )

                probability = max(
                    [
                        self._safe_probability(x)
                        for x in probability
                    ],
                    default=confidence
                )

            else:

                probability = self._safe_probability(
                    probability
                )

            result["probability"] = probability

            # ----------------------------------------------------
            # Uncertainty
            # ----------------------------------------------------

            uncertainty = result.get(
                "uncertainty",
                1.0 - confidence
            )

            result["uncertainty"] = (
                self._safe_probability(
                    uncertainty
                )
            )

            # ----------------------------------------------------
            # Quality
            # ----------------------------------------------------

            quality = result.get(
                "quality",
                1.0
            )

            result["quality"] = (
                self._safe_probability(
                    quality
                )
            )

            # ----------------------------------------------------
            # Missing data
            # ----------------------------------------------------

            missing_ratio = result.get(
                "missing_data_ratio",
                0.0
            )

            result["missing_data_ratio"] = (
                self._safe_probability(
                    missing_ratio
                )
            )

            # ----------------------------------------------------
            # Trust
            # ----------------------------------------------------

            result["trust"] = self._compute_trust(
                agent_name,
                result
            )

            self._log(
                f"[{agent_name.upper()}] ✓ Completed "
                f"({latency_ms:.2f} ms)"
            )

            return result

        except Exception as exc:

            latency_ms = (
                time.perf_counter() - start
            ) * 1000.0

            self._log(
                f"[{agent_name.upper()}] ✗ Error: {exc}"
            )

            return {
                "agent_id": agent_name,
                "agent": agent_name,
                "task_type": agent_name,
                "status": "error",
                "prediction": None,
                "probability": 0.0,
                "confidence": 0.0,
                "uncertainty": 1.0,
                "quality": 0.0,
                "trust": 0.0,
                "latency_ms": latency_ms,
                "missing_data_ratio": 1.0,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

    # ============================================================
    # TRUST
    # ============================================================

    def _compute_trust(
        self,
        agent_name,
        result
    ):

        confidence = self._safe_probability(
            result.get("confidence", 0.0)
        )

        quality = self._safe_probability(
            result.get("quality", 0.0)
        )

        uncertainty = self._safe_probability(
            result.get("uncertainty", 1.0)
        )

        missing_ratio = self._safe_probability(
            result.get(
                "missing_data_ratio",
                0.0
            )
        )

        if self.trust_manager is None:

            return max(
                0.0,
                min(
                    1.0,
                    confidence
                    * quality
                    * (1.0 - missing_ratio)
                    * (1.0 - 0.5 * uncertainty)
                )
            )

        try:

            return float(
                self.trust_manager.compute_trust(
                    agent_id=agent_name,
                    confidence=confidence,
                    quality=quality,
                    uncertainty=uncertainty,
                    missing_data_ratio=missing_ratio,
                    agreement=0.5,
                    stability=0.5,
                    utility=0.5,
                    modality_available=True,
                )
            )

        except Exception:

            return max(
                0.0,
                min(
                    1.0,
                    confidence
                    * quality
                    * (1.0 - missing_ratio)
                )
            )

    # ============================================================
    # SPECIALIZED AGENTS
    # ============================================================

    def run_fatty_liver(
        self,
        clinical_data
    ):

        return self._execute_agent(
            "fatty_liver",
            self.fatty_agent,
            clinical_data
        )

    def run_fibrosis(
        self,
        fibrosis_input
    ):

        return self._execute_agent(
            "fibrosis",
            self.fibrosis_agent,
            fibrosis_input
        )

    def run_cirrhosis(
        self,
        cirrhosis_input
    ):

        return self._execute_agent(
            "cirrhosis",
            self.cirrhosis_agent,
            cirrhosis_input
        )

    def run_tumor_classification(
        self,
        image
    ):

        return self._execute_agent(
            "tumor_classification",
            self.tumor_agent,
            image
        )

    def run_liver_segmentation(
        self,
        volume
    ):

        return self._execute_agent(
            "liver_segmentation",
            self.segmentation_agent,
            volume
        )

    # ============================================================
    # INPUT EXTRACTION
    # ============================================================

    def _extract_inputs(
        self,
        patient_data,
        clinical_data=None,
        fibrosis_input=None,
        cirrhosis_input=None,
        image=None,
        volume=None,
        clinical_reasoning_input=None
    ):

        patient_data = (
            patient_data
            if isinstance(patient_data, dict)
            else {}
        )

        # --------------------------------------------------------
        # Fatty liver
        # --------------------------------------------------------

        fatty_data = patient_data.get(
            "fatty_liver",
            clinical_data
        )

        # --------------------------------------------------------
        # Fibrosis
        # --------------------------------------------------------

        fibrosis_data = patient_data.get(
            "fibrosis",
            fibrosis_input
        )

        # --------------------------------------------------------
        # Cirrhosis
        # --------------------------------------------------------

        cirrhosis_data = patient_data.get(
            "cirrhosis",
            cirrhosis_input
        )

        # --------------------------------------------------------
        # Tumor
        # --------------------------------------------------------

        tumor_image = patient_data.get(
            "tumor",
            image
        )

        if tumor_image is None:

            tumor_image = patient_data.get(
                "tumor_classification",
                image
            )

        # --------------------------------------------------------
        # Segmentation
        # --------------------------------------------------------

        segmentation_volume = patient_data.get(
            "segmentation",
            volume
        )

        if segmentation_volume is None:

            segmentation_volume = patient_data.get(
                "liver_segmentation",
                volume
            )

        # --------------------------------------------------------
        # Clinical reasoning
        # --------------------------------------------------------

        reasoning_data = patient_data.get(
            "clinical_reasoning",
            clinical_reasoning_input
        )

        if reasoning_data is None:

            reasoning_data = fatty_data

        return {
            "fatty_liver": fatty_data,
            "fibrosis": fibrosis_data,
            "cirrhosis": cirrhosis_data,
            "tumor": tumor_image,
            "segmentation": segmentation_volume,
            "clinical_reasoning": reasoning_data,
        }

    # ============================================================
    # ADAPTIVE FUSION
    # ============================================================

    def _run_adaptive_fusion(
        self,
        specialized_results
    ):

        valid = []

        for result in specialized_results.values():

            if not isinstance(result, dict):
                continue

            if result.get("status") != "success":
                continue

            valid.append(result)

        if not valid:

            return {
                "status": "no_valid_results",
                "results": [],
            }

        if self.adaptive_fusion is None:

            return {
                "status": "fallback",
                "results": valid,
            }

        try:

            fused = self.adaptive_fusion.fuse(
                valid
            )

            return {
                "status": "success",
                "results": fused,
            }

        except Exception as exc:

            return {
                "status": "fallback",
                "error": str(exc),
                "results": valid,
            }

    # ============================================================
    # CONFLICT DETECTION
    # ============================================================

    def _run_conflict_detection(
        self,
        specialized_results
    ):

        if self.conflict_detector is None:

            return []

        if AgentResult is None:

            return []

        messages = []

        for result in specialized_results.values():

            if not isinstance(result, dict):
                continue

            if result.get("status") != "success":
                continue

            try:

                messages.append(
                    AgentResult.from_dict(result)
                )

            except Exception:

                continue

        if not messages:

            return []

        try:

            conflicts = (
                self.conflict_detector.detect(
                    messages
                )
            )

            return conflicts

        except Exception as exc:

            self._log(
                f"[CONFLICT] Warning: {exc}"
            )

            return []

    # ============================================================
    # CLINICAL REASONING
    # ============================================================

    def _run_clinical_reasoning(
        self,
        clinical_reasoning_input,
        specialized_results
    ):

        if self.clinical_reasoning_agent is None:

            return {
                "agent_id": "clinical_reasoning",
                "agent": "clinical_reasoning",
                "task_type": "clinical_reasoning",
                "status": "not_available",
                "prediction": None,
                "confidence": 0.0,
                "probability": 0.0,
                "uncertainty": 1.0,
                "quality": 0.0,
                "trust": 0.0,
            }

        # --------------------------------------------------------
        # IMPORTANT:
        # The trained Clinical Reasoning model expects the
        # BUPA six-feature input, therefore we do NOT send the
        # complete multimodal dictionary directly to the model.
        # --------------------------------------------------------

        result = self._execute_agent(
            "clinical_reasoning",
            self.clinical_reasoning_agent,
            clinical_reasoning_input
        )

        # Preserve specialized evidence
        result["specialized_evidence"] = (
            specialized_results
        )

        return result

    # ============================================================
    # DECISION ENGINE
    # ============================================================

    def _run_decision_engine(
        self,
        specialized_results,
        clinical_result,
        conflicts
    ):

        all_results = []

        for result in specialized_results.values():

            if isinstance(result, dict):
                all_results.append(result)

        if isinstance(
            clinical_result,
            dict
        ):

            all_results.append(
                clinical_result
            )

        # --------------------------------------------------------
        # Fallback decision
        # --------------------------------------------------------

        if self.decision_engine is None:

            successful = [
                r for r in all_results
                if r.get("status") == "success"
            ]

            return {
                "status": "fallback",
                "agents_available": len(
                    successful
                ),
                "agents_total": len(
                    all_results
                ),
                "conflicts": len(
                    conflicts or []
                ),
            }

        try:

            decision = self.decision_engine.decide(
                all_results,
                conflicts,
                clinical_result
            )

            return decision

        except Exception as exc:

            self._log(
                f"[DECISION] Warning: {exc}"
            )

            return {
                "status": "fallback",
                "error": str(exc),
                "agents_available": sum(
                    r.get("status") == "success"
                    for r in all_results
                ),
                "agents_total": len(
                    all_results
                ),
                "conflicts": len(
                    conflicts or []
                ),
            }

    # ============================================================
    # MAIN RUN
    # ============================================================

    def run(
        self,
        patient_id,
        patient_data=None,
        clinical_data=None,
        fibrosis_input=None,
        cirrhosis_input=None,
        image=None,
        volume=None,
        clinical_reasoning_input=None
    ):

        self.execution_log = []

        self._log("\n")
        self._log("=" * 80)
        self._log(
            "LIVERAI MULTI-AGENT ANALYSIS"
        )
        self._log("=" * 80)

        # --------------------------------------------------------
        # INPUTS
        # --------------------------------------------------------

        inputs = self._extract_inputs(
            patient_data=patient_data,
            clinical_data=clinical_data,
            fibrosis_input=fibrosis_input,
            cirrhosis_input=cirrhosis_input,
            image=image,
            volume=volume,
            clinical_reasoning_input=clinical_reasoning_input
        )

        # --------------------------------------------------------
        # STEP 1
        # --------------------------------------------------------

        self._log(
            "\nSTEP 1/5 → SPECIALIZED AGENTS"
        )

        specialized_results = {}

        specialized_results[
            "fatty_liver"
        ] = self.run_fatty_liver(
            inputs["fatty_liver"]
        )

        specialized_results[
            "fibrosis"
        ] = self.run_fibrosis(
            inputs["fibrosis"]
        )

        specialized_results[
            "cirrhosis"
        ] = self.run_cirrhosis(
            inputs["cirrhosis"]
        )

        specialized_results[
            "tumor_classification"
        ] = self.run_tumor_classification(
            inputs["tumor"]
        )

        specialized_results[
            "liver_segmentation"
        ] = self.run_liver_segmentation(
            inputs["segmentation"]
        )

        # --------------------------------------------------------
        # STEP 2
        # --------------------------------------------------------

        self._log(
            "\nSTEP 2/5 → ADAPTIVE FUSION"
        )

        fusion_result = (
            self._run_adaptive_fusion(
                specialized_results
            )
        )

        # --------------------------------------------------------
        # STEP 3
        # --------------------------------------------------------

        self._log(
            "\nSTEP 3/5 → CONFLICT DETECTION"
        )

        conflicts = (
            self._run_conflict_detection(
                specialized_results
            )
        )

        # --------------------------------------------------------
        # STEP 4
        # --------------------------------------------------------

        self._log(
            "\nSTEP 4/5 → CLINICAL REASONING"
        )

        clinical_result = (
            self._run_clinical_reasoning(
                inputs["clinical_reasoning"],
                specialized_results
            )
        )

        # --------------------------------------------------------
        # STEP 5
        # --------------------------------------------------------

        self._log(
            "\nSTEP 5/5 → DECISION ENGINE"
        )

        decision = self._run_decision_engine(
            specialized_results,
            clinical_result,
            conflicts
        )

        # --------------------------------------------------------
        # Statistics
        # --------------------------------------------------------

        total_specialized = len(
            specialized_results
        )

        completed_specialized = sum(
            result.get("status") == "success"
            for result in specialized_results.values()
            if isinstance(result, dict)
        )

        total_agents = 6

        completed_agents = (
            completed_specialized
            +
            int(
                isinstance(
                    clinical_result,
                    dict
                )
                and clinical_result.get(
                    "status"
                ) == "success"
            )
        )

        if completed_agents == total_agents:

            system_status = "completed"

        elif completed_agents > 0:

            system_status = "partial"

        else:

            system_status = "failed"

        # --------------------------------------------------------
        # Final assessment
        # --------------------------------------------------------

        assessment = {
            "system": "LiverAI-MultiAgent",
            "patient_id": patient_id,
            "timestamp": datetime.now().isoformat(),

            "status": system_status,

            "agents_completed": completed_agents,
            "total_agents": total_agents,

            "specialized_agents_completed": (
                completed_specialized
            ),

            "total_specialized_agents": (
                total_specialized
            ),

            "agents": specialized_results,

            "clinical_reasoning": clinical_result,

            "fusion": fusion_result,

            "conflicts": conflicts,

            "decision": decision,

            "execution_log": self.execution_log,
        }

        self.last_results = {
            **specialized_results,
            "clinical_reasoning": clinical_result,
        }

        self.last_assessment = assessment

        self._log("\n")
        self._log("=" * 80)
        self._log(
            "✓ UNIFIED LIVER ASSESSMENT GENERATED"
        )
        self._log(
            f"System status: {system_status}"
        )
        self._log(
            f"Agents completed: "
            f"{completed_agents}/{total_agents}"
        )
        self._log("=" * 80)

        return assessment

    # ============================================================
    # ANALYZE ALIAS
    # ============================================================

    def analyze(
        self,
        patient_id,
        patient_data=None,
        **kwargs
    ):

        return self.run(
            patient_id=patient_id,
            patient_data=patient_data,
            **kwargs
        )

    # ============================================================
    # BACKWARD-COMPATIBLE PREDICT
    # ============================================================

    def predict(
        self,
        clinical_data=None,
        ultrasound_image=None,
        mri_image=None,
        liver_volume=None,
        patient_id="unknown_patient",
        **kwargs
    ):

        patient_data = {
            "fatty_liver": clinical_data,
            "fibrosis": (
                kwargs.get(
                    "fibrosis_input",
                    ultrasound_image
                )
            ),
            "cirrhosis": (
                kwargs.get(
                    "cirrhosis_input",
                    clinical_data
                )
            ),
            "tumor": mri_image,
            "segmentation": liver_volume,
            "clinical_reasoning": clinical_data,
        }

        return self.run(
            patient_id=patient_id,
            patient_data=patient_data
        )

    # ============================================================
    # GETTERS
    # ============================================================

    def get_last_results(self):

        return self.last_results

    def get_last_assessment(self):

        return self.last_assessment

    def get_execution_log(self):

        return self.execution_log

    # ============================================================
    # HEALTH CHECK
    # ============================================================

    def health_check(self):

        result = {}

        for name, agent in self.agents.items():

            result[name] = {
                "loaded": agent is not None,
                "class": (
                    agent.__class__.__name__
                    if agent is not None
                    else None
                ),
            }

        return result

    # Backward-compatible name
    get_system_status = health_check
```
