# ============================================================
# LiverAI Multi-Agent Orchestrator
# ============================================================
#
# Coordinates:
#   1. Fatty Liver Agent
#   2. Fibrosis Agent
#   3. Cirrhosis Agent
#   4. Tumor Classification Agent
#   5. Liver Segmentation Agent
#   6. Clinical Reasoning Agent
#
# ============================================================

from typing import Dict, Any, Optional
import os
import traceback


# ============================================================
# AGENTS
# ============================================================

try:
    from agents.fatty_liver_agent import FattyLiverAgent
except Exception:
    FattyLiverAgent = None

try:
    from agents.fibrosis_agent import FibrosisAgent
except Exception:
    FibrosisAgent = None

try:
    from agents.cirrhosis_agent import CirrhosisAgent
except Exception:
    CirrhosisAgent = None

try:
    from agents.tumor_classification_agent import TumorClassificationAgent
except Exception:
    TumorClassificationAgent = None

try:
    from agents.liver_segmentation_agent import LiverSegmentationAgent
except Exception:
    LiverSegmentationAgent = None

try:
    from agents.clinical_reasoning_agent import ClinicalReasoningAgent
except Exception:
    ClinicalReasoningAgent = None


# ============================================================
# COORDINATION MODULES
# ============================================================

try:
    from communication.agent_result import AgentResult
except Exception:
    AgentResult = None

try:
    from coordinator.trust_manager import TrustManager
except Exception:
    TrustManager = None

try:
    from coordinator.adaptive_fusion import AdaptiveFusion
except Exception:
    AdaptiveFusion = None

try:
    from coordinator.conflict_detector import ConflictDetector
except Exception:
    ConflictDetector = None

try:
    from coordinator.decision_engine import DecisionEngine
except Exception:
    DecisionEngine = None


# ============================================================
# MODEL PATHS
# ============================================================

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


# ============================================================
# ORCHESTRATOR
# ============================================================

class LiverAIOrchestrator:

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        fatty_agent=None,
        fibrosis_agent=None,
        cirrhosis_agent=None,
        tumor_agent=None,
        segmentation_agent=None,
        clinical_reasoning_agent=None,
        fatty_liver_agent=None,
        tumor_classification_agent=None,
        liver_segmentation_agent=None,
        clinical_agent=None,
        auto_initialize=True,
        device=None,
    ):

        self.device = device

        # ----------------------------------------------------
        # Accept both naming conventions
        # ----------------------------------------------------

        self.fatty_agent = (
            fatty_agent
            if fatty_agent is not None
            else fatty_liver_agent
        )

        self.fibrosis_agent = fibrosis_agent

        self.cirrhosis_agent = cirrhosis_agent

        self.tumor_agent = (
            tumor_agent
            if tumor_agent is not None
            else tumor_classification_agent
        )

        self.segmentation_agent = (
            segmentation_agent
            if segmentation_agent is not None
            else liver_segmentation_agent
        )

        self.clinical_reasoning_agent = (
            clinical_reasoning_agent
            if clinical_reasoning_agent is not None
            else clinical_agent
        )

        # ----------------------------------------------------
        # Coordination modules
        # ----------------------------------------------------

        self.trust_manager = None
        self.adaptive_fusion = None
        self.conflict_detector = None
        self.decision_engine = None

        self.initialization_errors = {}

        # ----------------------------------------------------
        # Initialize agents if requested
        # ----------------------------------------------------

        if auto_initialize:
            self._initialize_agents()

        # ----------------------------------------------------
        # Initialize coordination components
        # ----------------------------------------------------

        self._initialize_coordination_modules()

    # ========================================================
    # INITIALIZE AGENTS
    # ========================================================

    def _initialize_agents(self):

        # ----------------------------------------------------
        # FATty liver
        # ----------------------------------------------------

        if self.fatty_agent is None:

            try:
                if FattyLiverAgent is None:
                    raise ImportError(
                        "FattyLiverAgent could not be imported."
                    )

                import joblib

                model = joblib.load(FATTY_MODEL_PATH)

                self.fatty_agent = FattyLiverAgent(
                    model=model
                )

                print("✓ Fatty Liver Agent initialized")

            except Exception as e:

                self.initialization_errors["fatty_liver"] = str(e)

                print(
                    "✗ Fatty Liver Agent initialization failed:",
                    e
                )

        # ----------------------------------------------------
        # FIBROSIS
        # ----------------------------------------------------

        if self.fibrosis_agent is None:

            try:
                if FibrosisAgent is None:
                    raise ImportError(
                        "FibrosisAgent could not be imported."
                    )

                import joblib

                model = joblib.load(FIBROSIS_MODEL_PATH)

                self.fibrosis_agent = FibrosisAgent(
                    model=model
                )

                print("✓ Fibrosis Agent initialized")

            except Exception as e:

                self.initialization_errors["fibrosis"] = str(e)

                print(
                    "✗ Fibrosis Agent initialization failed:",
                    e
                )

        # ----------------------------------------------------
        # CIRRHOSIS
        # ----------------------------------------------------

        if self.cirrhosis_agent is None:

            try:
                if CirrhosisAgent is None:
                    raise ImportError(
                        "CirrhosisAgent could not be imported."
                    )

                self.cirrhosis_agent = CirrhosisAgent(
                    model_path=CIRRHOSIS_MODEL_PATH
                )

                print("✓ Cirrhosis Agent initialized")

            except Exception as e:

                self.initialization_errors["cirrhosis"] = str(e)

                print(
                    "✗ Cirrhosis Agent initialization failed:",
                    e
                )

        # ----------------------------------------------------
        # TUMOR CLASSIFICATION
        # ----------------------------------------------------

        if self.tumor_agent is None:

            try:
                if TumorClassificationAgent is None:
                    raise ImportError(
                        "TumorClassificationAgent "
                        "could not be imported."
                    )

                self.tumor_agent = TumorClassificationAgent(
                    model_path=TUMOR_MODEL_PATH
                )

                print(
                    "✓ Tumor Classification Agent initialized"
                )

            except Exception as e:

                self.initialization_errors[
                    "tumor_classification"
                ] = str(e)

                print(
                    "✗ Tumor Classification Agent "
                    "initialization failed:",
                    e
                )

        # ----------------------------------------------------
        # LIVER SEGMENTATION
        # ----------------------------------------------------

        if self.segmentation_agent is None:

            try:
                if LiverSegmentationAgent is None:
                    raise ImportError(
                        "LiverSegmentationAgent "
                        "could not be imported."
                    )

                self.segmentation_agent = (
                    LiverSegmentationAgent(
                        model_path=SEGMENTATION_MODEL_PATH
                    )
                )

                print(
                    "✓ Liver Segmentation Agent initialized"
                )

            except Exception as e:

                self.initialization_errors[
                    "liver_segmentation"
                ] = str(e)

                print(
                    "✗ Liver Segmentation Agent "
                    "initialization failed:",
                    e
                )

        # ----------------------------------------------------
        # CLINICAL REASONING
        # ----------------------------------------------------

        if self.clinical_reasoning_agent is None:

            try:
                if ClinicalReasoningAgent is None:
                    raise ImportError(
                        "ClinicalReasoningAgent "
                        "could not be imported."
                    )

                self.clinical_reasoning_agent = (
                    ClinicalReasoningAgent(
                        CLINICAL_MODEL_PATH
                    )
                )

                print(
                    "✓ Clinical Reasoning Agent initialized"
                )

            except Exception as e:

                self.initialization_errors[
                    "clinical_reasoning"
                ] = str(e)

                print(
                    "✗ Clinical Reasoning Agent "
                    "initialization failed:",
                    e
                )

    # ========================================================
    # INITIALIZE COORDINATION MODULES
    # ========================================================

    def _initialize_coordination_modules(self):

        # ----------------------------------------------------
        # TRUST MANAGER
        # ----------------------------------------------------

        if TrustManager is not None:

            try:
                self.trust_manager = TrustManager()
            except Exception:
                try:
                    self.trust_manager = TrustManager
                except Exception:
                    self.trust_manager = None

        # ----------------------------------------------------
        # ADAPTIVE FUSION
        # ----------------------------------------------------

        if AdaptiveFusion is not None:

            try:
                self.adaptive_fusion = AdaptiveFusion()
            except Exception:
                try:
                    self.adaptive_fusion = AdaptiveFusion
                except Exception:
                    self.adaptive_fusion = None

        # ----------------------------------------------------
        # CONFLICT DETECTOR
        # ----------------------------------------------------

        if ConflictDetector is not None:

            try:
                self.conflict_detector = ConflictDetector()
            except Exception:
                try:
                    self.conflict_detector = ConflictDetector
                except Exception:
                    self.conflict_detector = None

        # ----------------------------------------------------
        # DECISION ENGINE
        # ----------------------------------------------------

        if DecisionEngine is not None:

            try:
                self.decision_engine = DecisionEngine()
            except Exception:
                try:
                    self.decision_engine = DecisionEngine
                except Exception:
                    self.decision_engine = None

    # ========================================================
    # MAIN RUN METHOD
    # ========================================================

    def run(
        self,
        patient_id: str = "UNKNOWN",
        patient_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter() 
        # ----------------------------------------------------
        # Validate patient data
        # ----------------------------------------------------

        if patient_data is None:
            patient_data = {}

        # ----------------------------------------------------
        # 1. SPECIALIZED AGENTS
        # ----------------------------------------------------

        specialized_results = self.run_specialized_agents(
            patient_data
        )

        # ----------------------------------------------------
        # 2. CLINICAL REASONING
        # ----------------------------------------------------

        clinical_result = self.run_clinical_reasoning(
            patient_data,
            specialized_results,
        )

        if clinical_result is None:

            clinical_result = {
                "status": "error",
                "error": "Clinical reasoning failed",
            }

        # ----------------------------------------------------
        # 3. COMBINE ALL RESULTS
        # ----------------------------------------------------

        all_results = dict(specialized_results)

        all_results["clinical_reasoning"] = clinical_result

        # ----------------------------------------------------
        # 4. ADAPTIVE FUSION
        # ----------------------------------------------------

        fusion = self._run_adaptive_fusion(
            all_results
        )

        # ----------------------------------------------------
        # 5. CONFLICT DETECTION
        # ----------------------------------------------------

        conflicts = self._run_conflict_detection(
            all_results
        )

        # ----------------------------------------------------
        # 6. DECISION ENGINE
        # ----------------------------------------------------

        decision = self._run_decision_engine(
            results=all_results,
            conflicts=conflicts,
            fusion=fusion,
            clinical_result=clinical_result,
        )

        # ----------------------------------------------------
        # 7. FINAL STATUS
        # ----------------------------------------------------

        successful_agents = 0
        total_agents = len(all_results)

        for result in all_results.values():

            if isinstance(result, dict):

                if result.get("status") == "success":
                    successful_agents += 1

        if total_agents == 0:

            overall_status = "error"

        elif successful_agents == total_agents:

            overall_status = "success"

        elif successful_agents > 0:

            overall_status = "partial"

        else:

            overall_status = "error"

        # ----------------------------------------------------
        # 8. FINAL OUTPUT
        # ----------------------------------------------------

        return {
            "status": overall_status,
            "patient_id": patient_id,
            "agents": all_results,
            "fusion": fusion,
            "conflicts": conflicts,
            "decision": decision,
            "summary": {
                "total_agents": total_agents,
                "successful_agents": successful_agents,
                "failed_agents": (
                    total_agents - successful_agents
                ),
            },
        }

    # ========================================================
    # ALIASES
    # ========================================================

    def analyze(
        self,
        patient_id: str = "UNKNOWN",
        patient_data: Optional[Dict[str, Any]] = None,
    ):

        return self.run(
            patient_id=patient_id,
            patient_data=patient_data,
        )

    def predict(
        self,
        patient_id: str = "UNKNOWN",
        patient_data: Optional[Dict[str, Any]] = None,
    ):

        return self.run(
            patient_id=patient_id,
            patient_data=patient_data,
        )

    # ========================================================
    # SPECIALIZED AGENTS
    # ========================================================

    def run_specialized_agents(
        self,
        patient_data: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:

        results = {}

        # ----------------------------------------------------
        # FATty liver
        # ----------------------------------------------------

        fatty_data = patient_data.get(
            "fatty_liver",
            patient_data.get("fatty", {})
        )

        results["fatty_liver"] = self._execute_agent(
            agent=self.fatty_agent,
            data=fatty_data,
            agent_name="fatty_liver",
        )

        # ----------------------------------------------------
        # FIBROSIS
        # ----------------------------------------------------

        fibrosis_data = patient_data.get(
            "fibrosis",
            {}
        )

        results["fibrosis"] = self._execute_agent(
            agent=self.fibrosis_agent,
            data=fibrosis_data,
            agent_name="fibrosis",
        )

        # ----------------------------------------------------
        # CIRRHOSIS
        # ----------------------------------------------------

        cirrhosis_data = patient_data.get(
            "cirrhosis",
            {}
        )

        results["cirrhosis"] = self._execute_agent(
            agent=self.cirrhosis_agent,
            data=cirrhosis_data,
            agent_name="cirrhosis",
        )

        # ----------------------------------------------------
        # TUMOR
        # ----------------------------------------------------

        tumor_data = patient_data.get(
            "tumor_classification",
            patient_data.get("tumor", None)
        )

        results["tumor_classification"] = (
            self._execute_agent(
                agent=self.tumor_agent,
                data=tumor_data,
                agent_name="tumor_classification",
            )
        )

        # ----------------------------------------------------
        # SEGMENTATION
        # ----------------------------------------------------

        segmentation_data = patient_data.get(
            "liver_segmentation",
            patient_data.get("segmentation", None)
        )

        results["liver_segmentation"] = (
            self._execute_agent(
                agent=self.segmentation_agent,
                data=segmentation_data,
                agent_name="liver_segmentation",
            )
        )

        return results

    # ========================================================
    # CLINICAL REASONING
    # ========================================================

    def run_clinical_reasoning(
        self,
        patient_data: Dict[str, Any],
        specialized_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:

        if self.clinical_reasoning_agent is None:

            return {
                "status": "error",
                "error": (
                    "Clinical reasoning agent "
                    "is not initialized"
                ),
            }

        # ----------------------------------------------------
        # Prepare clinical input
        # ----------------------------------------------------

        clinical_data = patient_data.get(
            "clinical_reasoning",
            {}
        )

        # ----------------------------------------------------
        # If clinical data is empty, try fatty-liver data
        # ----------------------------------------------------

        if not clinical_data:

            clinical_data = patient_data.get(
                "fatty_liver",
                {}
            )

        # ----------------------------------------------------
        # Add specialized results
        # ----------------------------------------------------

        clinical_context = self._build_clinical_context(
            clinical_data=clinical_data,
            specialized_results=specialized_results,
        )

        # ----------------------------------------------------
        # Execute
        # ----------------------------------------------------

        try:

            result = self._call_agent(
                self.clinical_reasoning_agent,
                clinical_context,
            )

            return self._normalize_result(
                result,
                agent_name="clinical_reasoning",
            )

        except Exception as e:

            return {
                "status": "error",
                "agent": "clinical_reasoning",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    # ========================================================
    # EXECUTE AGENT
    # ========================================================

    def _execute_agent(
        self,
        agent,
        data,
        agent_name: str,
    ) -> Dict[str, Any]:

        if agent is None:

            return {
                "status": "error",
                "agent": agent_name,
                "error": "Agent is not initialized",
            }

        if data is None:

            return {
                "status": "not_run",
                "agent": agent_name,
                "error": "No input data provided",
            }

        try:

            result = self._call_agent(
                agent,
                data,
            )

            return self._normalize_result(
                result,
                agent_name=agent_name,
            )

        except Exception as e:

            return {
                "status": "error",
                "agent": agent_name,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    # ========================================================
    # CALL AGENT
    # ========================================================

    def _call_agent(
        self,
        agent,
        data,
    ):

        # ----------------------------------------------------
        # predict()
        # ----------------------------------------------------

        if hasattr(agent, "predict"):

            try:
                return agent.predict(data)
            except TypeError:

                try:
                    return agent.predict(
                        patient_data=data
                    )
                except TypeError:
                    pass

        # ----------------------------------------------------
        # run()
        # ----------------------------------------------------

        if hasattr(agent, "run"):

            try:
                return agent.run(data)
            except TypeError:

                try:
                    return agent.run(
                        patient_data=data
                    )
                except TypeError:
                    pass

        # ----------------------------------------------------
        # analyze()
        # ----------------------------------------------------

        if hasattr(agent, "analyze"):

            try:
                return agent.analyze(data)
            except TypeError:

                try:
                    return agent.analyze(
                        patient_data=data
                    )
                except TypeError:
                    pass

        raise AttributeError(
            f"Agent {type(agent).__name__} has no "
            "compatible predict/run/analyze method"
        )

    # ========================================================
    # NORMALIZE RESULT
    # ========================================================

    def _normalize_result(
        self,
        result,
        agent_name: str,
    ) -> Dict[str, Any]:

        # ----------------------------------------------------
        # AgentResult object
        # ----------------------------------------------------

        if AgentResult is not None:

            if isinstance(result, AgentResult):

                try:

                    if hasattr(result, "to_dict"):
                        result = result.to_dict()

                except Exception:
                    pass

        # ----------------------------------------------------
        # None
        # ----------------------------------------------------

        if result is None:

            return {
                "status": "success",
                "agent": agent_name,
                "prediction": None,
                "confidence": None,
                "uncertainty": None,
            }

        # ----------------------------------------------------
        # Dictionary
        # ----------------------------------------------------

        if isinstance(result, dict):

            normalized = dict(result)

        else:

            # ------------------------------------------------
            # Scalar / object prediction
            # ------------------------------------------------

            normalized = {
                "prediction": result
            }

        # ----------------------------------------------------
        # Default status
        # ----------------------------------------------------

        if "status" not in normalized:

            normalized["status"] = "success"

        normalized["agent"] = agent_name

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        if normalized.get("confidence") is None:

            confidence = self._extract_confidence(
                normalized
            )

            if confidence is not None:
                normalized["confidence"] = confidence

        # ----------------------------------------------------
        # Uncertainty
        # ----------------------------------------------------

        if normalized.get("uncertainty") is None:

            confidence = normalized.get(
                "confidence"
            )

            if confidence is not None:

                try:

                    normalized["uncertainty"] = (
                        1.0 - float(confidence)
                    )

                except Exception:
                    pass

        # ----------------------------------------------------
        # Quality
        # ----------------------------------------------------

        if "quality" not in normalized:

            normalized["quality"] = None

        # ----------------------------------------------------
        # Trust
        # ----------------------------------------------------

        if "trust" not in normalized:

            normalized["trust"] = self._compute_trust(
                agent_name=agent_name,
                result=normalized,
            )

        return normalized

    # ========================================================
    # EXTRACT CONFIDENCE
    # ========================================================

    def _extract_confidence(
        self,
        result: Dict[str, Any],
    ) -> Optional[float]:

        possible_keys = [
            "confidence",
            "probability",
            "score",
            "prob",
            "max_probability",
        ]

        for key in possible_keys:

            if key in result:

                value = result[key]

                try:

                    return self._clip(
                        float(value),
                        0.0,
                        1.0,
                    )

                except Exception:
                    pass

        # ----------------------------------------------------
        # probabilities
        # ----------------------------------------------------

        probabilities = result.get(
            "probabilities"
        )

        if probabilities is not None:

            return self._probability_confidence(
                probabilities
            )

        # ----------------------------------------------------
        # probs
        # ----------------------------------------------------

        probabilities = result.get(
            "probs"
        )

        if probabilities is not None:

            return self._probability_confidence(
                probabilities
            )

        return None

    # ========================================================
    # PROBABILITY CONFIDENCE
    # ========================================================

    def _probability_confidence(
        self,
        probabilities,
    ) -> Optional[float]:

        try:

            if isinstance(
                probabilities,
                dict
            ):

                values = probabilities.values()

                values = [
                    float(v)
                    for v in values
                ]

                if values:
                    return max(values)

            if isinstance(
                probabilities,
                (list, tuple)
            ):

                values = [
                    float(v)
                    for v in probabilities
                ]

                if values:
                    return max(values)

        except Exception:
            return None

        return None

    # ========================================================
    # COMPUTE TRUST
    # ========================================================

    def _compute_trust(
        self,
        agent_name: str,
        result: Dict[str, Any],
    ) -> float:

        confidence = result.get(
            "confidence"
        )

        quality = result.get(
            "quality"
        )

        # ----------------------------------------------------
        # Default trust
        # ----------------------------------------------------

        if confidence is None:
            confidence = 0.5

        if quality is None:
            quality = 1.0

        try:

            confidence = float(confidence)

        except Exception:

            confidence = 0.5

        try:

            quality = float(quality)

        except Exception:

            quality = 1.0

        confidence = self._clip(
            confidence,
            0.0,
            1.0,
        )

        quality = self._clip(
            quality,
            0.0,
            1.0,
        )

        # ----------------------------------------------------
        # TrustManager
        # ----------------------------------------------------

        if self.trust_manager is not None:

            try:

                if hasattr(
                    self.trust_manager,
                    "compute_trust",
                ):

                    trust = (
                        self.trust_manager.compute_trust(
                            agent_name=agent_name,
                            confidence=confidence,
                            quality=quality,
                            result=result,
                        )
                    )

                    return self._clip(
                        float(trust),
                        0.0,
                        1.0,
                    )

            except Exception:
                pass

            try:

                if hasattr(
                    self.trust_manager,
                    "get_trust",
                ):

                    trust = (
                        self.trust_manager.get_trust(
                            agent_name
                        )
                    )

                    return self._clip(
                        float(trust),
                        0.0,
                        1.0,
                    )

            except Exception:
                pass

        # ----------------------------------------------------
        # Simple fallback
        # ----------------------------------------------------

        return self._clip(
            0.7 * confidence
            + 0.3 * quality,
            0.0,
            1.0,
        )

    # ========================================================
    # ADAPTIVE FUSION
    # ========================================================

    def _run_adaptive_fusion(
        self,
        results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:

        successful_results = {}

        for name, result in results.items():

            if not isinstance(result, dict):
                continue

            if result.get("status") != "success":
                continue

            if result.get("prediction") is None:
                continue

            successful_results[name] = result

        # ----------------------------------------------------
        # Nothing to fuse
        # ----------------------------------------------------

        if not successful_results:

            return {
                "status": "not_available",
                "prediction": None,
                "confidence": None,
                "weights": {},
            }

        # ----------------------------------------------------
        # AdaptiveFusion
        # ----------------------------------------------------

        if self.adaptive_fusion is not None:

            try:

                if hasattr(
                    self.adaptive_fusion,
                    "fuse",
                ):

                    fusion = (
                        self.adaptive_fusion.fuse(
                            successful_results
                        )
                    )

                    if isinstance(
                        fusion,
                        dict,
                    ):

                        return fusion

                    return {
                        "status": "success",
                        "prediction": fusion,
                    }

            except Exception:
                pass

            try:

                if hasattr(
                    self.adaptive_fusion,
                    "combine",
                ):

                    fusion = (
                        self.adaptive_fusion.combine(
                            successful_results
                        )
                    )

                    if isinstance(
                        fusion,
                        dict,
                    ):

                        return fusion

                    return {
                        "status": "success",
                        "prediction": fusion,
                    }

            except Exception:
                pass

        # ----------------------------------------------------
        # Weighted fallback
        # ----------------------------------------------------

        weighted_scores = {}
        total_weight = 0.0

        for name, result in successful_results.items():

            prediction = result.get(
                "prediction"
            )

            confidence = result.get(
                "confidence"
            )

            trust = result.get(
                "trust",
                1.0,
            )

            try:

                weight = float(
                    confidence
                    if confidence is not None
                    else 0.5
                )

            except Exception:

                weight = 0.5

            try:

                trust = float(trust)

            except Exception:

                trust = 1.0

            weight *= self._clip(
                trust,
                0.0,
                1.0,
            )

            try:

                numeric_prediction = float(
                    prediction
                )

            except Exception:

                continue

            weighted_scores[name] = {
                "prediction": numeric_prediction,
                "weight": weight,
            }

            total_weight += weight

        if total_weight <= 0:

            return {
                "status": "success",
                "prediction": None,
                "confidence": None,
                "weights": {},
            }

        weighted_prediction = sum(
            item["prediction"] * item["weight"]
            for item in weighted_scores.values()
        ) / total_weight

        return {
            "status": "success",
            "prediction": weighted_prediction,
            "confidence": self._clip(
                total_weight
                / max(
                    len(weighted_scores),
                    1,
                ),
                0.0,
                1.0,
            ),
            "weights": {
                name: item["weight"]
                for name, item
                in weighted_scores.items()
            },
        }

    # ========================================================
    # CONFLICT DETECTION
    # ========================================================

    def _run_conflict_detection(
        self,
        results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:

        valid_results = {}

        for name, result in results.items():

            if not isinstance(result, dict):
                continue

            if result.get("status") != "success":
                continue

            if result.get("prediction") is None:
                continue

            valid_results[name] = result

        # ----------------------------------------------------
        # ConflictDetector
        # ----------------------------------------------------

        if self.conflict_detector is not None:

            try:

                if hasattr(
                    self.conflict_detector,
                    "detect",
                ):

                    conflicts = (
                        self.conflict_detector.detect(
                            valid_results
                        )
                    )

                    if isinstance(
                        conflicts,
                        dict,
                    ):

                        return conflicts

                    return {
                        "status": "success",
                        "has_conflict": bool(
                            conflicts
                        ),
                        "conflicts": conflicts,
                    }

            except Exception:
                pass

            try:

                if hasattr(
                    self.conflict_detector,
                    "check_conflicts",
                ):

                    conflicts = (
                        self.conflict_detector.check_conflicts(
                            valid_results
                        )
                    )

                    if isinstance(
                        conflicts,
                        dict,
                    ):

                        return conflicts

                    return {
                        "status": "success",
                        "has_conflict": bool(
                            conflicts
                        ),
                        "conflicts": conflicts,
                    }

            except Exception:
                pass

        # ----------------------------------------------------
        # Fallback conflict detection
        # ----------------------------------------------------

        predictions = []

        for name, result in valid_results.items():

            predictions.append(
                (
                    name,
                    result.get("prediction"),
                )
            )

        if len(predictions) < 2:

            return {
                "status": "success",
                "has_conflict": False,
                "conflicts": [],
            }

        numeric_predictions = []

        for name, prediction in predictions:

            try:

                numeric_predictions.append(
                    (
                        name,
                        float(prediction),
                    )
                )

            except Exception:
                pass

        if len(numeric_predictions) < 2:

            return {
                "status": "success",
                "has_conflict": False,
                "conflicts": [],
            }

        values = [
            value
            for _, value
            in numeric_predictions
        ]

        min_value = min(values)
        max_value = max(values)

        # Difference greater than 1 means
        # potentially different clinical outputs.
        has_conflict = (
            abs(max_value - min_value) > 1.0
        )

        return {
            "status": "success",
            "has_conflict": has_conflict,
            "conflicts": (
                numeric_predictions
                if has_conflict
                else []
            ),
        }

    # ========================================================
    # DECISION ENGINE
    # ========================================================

    def _run_decision_engine(
        self,
        results: Dict[str, Dict[str, Any]],
        conflicts: Dict[str, Any],
        fusion: Dict[str, Any],
        clinical_result: Dict[str, Any],
    ) -> Dict[str, Any]:

        # ----------------------------------------------------
        # DecisionEngine
        # ----------------------------------------------------

        if self.decision_engine is not None:

            try:

                if hasattr(
                    self.decision_engine,
                    "decide",
                ):

                    result_list = list(
                        results.values()
                    )

                    decision = (
                        self.decision_engine.decide(
                            results=result_list,
                            conflicts=conflicts,
                            reasoning=clinical_result,
                        )
                    )

                    if isinstance(
                        decision,
                        dict,
                    ):

                        return decision

                    return {
                        "status": "success",
                        "decision": decision,
                    }

            except Exception:
                pass

        # ----------------------------------------------------
        # Fallback decision
        # ----------------------------------------------------

        clinical_prediction = (
            clinical_result.get(
                "prediction"
            )
            if isinstance(
                clinical_result,
                dict,
            )
            else None
        )

        fusion_prediction = (
            fusion.get(
                "prediction"
            )
            if isinstance(
                fusion,
                dict,
            )
            else None
        )

        has_conflict = (
            conflicts.get(
                "has_conflict",
                False,
            )
            if isinstance(
                conflicts,
                dict,
            )
            else False
        )

        # ----------------------------------------------------
        # Clinical reasoning has priority
        # ----------------------------------------------------

        if clinical_prediction is not None:

            final_prediction = clinical_prediction

            source = "clinical_reasoning"

        elif fusion_prediction is not None:

            final_prediction = fusion_prediction

            source = "adaptive_fusion"

        else:

            final_prediction = None

            source = "none"

        return {
            "status": "success",
            "prediction": final_prediction,
            "source": source,
            "has_conflict": has_conflict,
            "confidence": self._extract_confidence(
                clinical_result
            ),
        }

    # ========================================================
    # BUILD FINAL DECISION
    # ========================================================

    def _build_final_decision(
        self,
        results: Dict[str, Dict[str, Any]],
        fusion: Dict[str, Any],
        conflicts: Dict[str, Any],
        clinical_result: Dict[str, Any],
    ) -> Dict[str, Any]:

        return self._run_decision_engine(
            results=results,
            conflicts=conflicts,
            fusion=fusion,
            clinical_result=clinical_result,
        )

    # ========================================================
    # NOT RUN RESULT
    # ========================================================

    def _not_run_result(
        self,
        agent_name: str,
        reason: str,
    ) -> Dict[str, Any]:

        return {
            "status": "not_run",
            "agent": agent_name,
            "prediction": None,
            "confidence": None,
            "uncertainty": None,
            "quality": None,
            "trust": 0.0,
            "error": reason,
        }

    # ========================================================
    # BUILD CLINICAL CONTEXT
    # ========================================================

    def _build_clinical_context(
        self,
        clinical_data: Dict[str, Any],
        specialized_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:

        context = {}

        if isinstance(
            clinical_data,
            dict,
        ):

            context.update(
                clinical_data
            )

        # ----------------------------------------------------
        # Add predictions
        # ----------------------------------------------------

        for agent_name, result in (
            specialized_results.items()
        ):

            if not isinstance(
                result,
                dict,
            ):
                continue

            prediction = result.get(
                "prediction"
            )

            confidence = result.get(
                "confidence"
            )

            context[
                f"{agent_name}_prediction"
            ] = prediction

            context[
                f"{agent_name}_confidence"
            ] = confidence

        return context

    # ========================================================
    # SAFE FLOAT
    # ========================================================

    def _safe_float(
        self,
        value,
        default: float = 0.0,
    ) -> float:

        try:
            return float(value)

        except Exception:
            return default

    # ========================================================
    # CLIP
    # ========================================================

    def _clip(
        self,
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:

        return max(
            minimum,
            min(
                maximum,
                value,
            ),
        )

    # ========================================================
    # HEALTH CHECK
    # ========================================================

    def health_check(self) -> Dict[str, Any]:

        agents = {
            "fatty_liver": self.fatty_agent,
            "fibrosis": self.fibrosis_agent,
            "cirrhosis": self.cirrhosis_agent,
            "tumor_classification": self.tumor_agent,
            "liver_segmentation": self.segmentation_agent,
            "clinical_reasoning": (
                self.clinical_reasoning_agent
            ),
        }

        agent_status = {}

        for name, agent in agents.items():

            agent_status[name] = (
                agent is not None
            )

        ready_agents = sum(
            1
            for value in agent_status.values()
            if value
        )

        coordinators = {
            "trust_manager": self.trust_manager,
            "adaptive_fusion": self.adaptive_fusion,
            "conflict_detector": self.conflict_detector,
            "decision_engine": self.decision_engine,
        }

        coordinator_status = {}

        for name, module in coordinators.items():

            coordinator_status[name] = (
                module is not None
            )

        ready_coordinators = sum(
            1
            for value in coordinator_status.values()
            if value
        )

        return {
            "status": (
                "healthy"
                if ready_agents == 6
                else "partial"
            ),
            "agents": agent_status,
            "agents_ready": ready_agents,
            "agents_total": 6,
            "coordinators": coordinator_status,
            "coordinators_ready": ready_coordinators,
            "coordinators_total": 4,
            "initialization_errors": (
                self.initialization_errors
            ),
        }

    # ========================================================
    # GETTERS
    # ========================================================

    def get_agents(self) -> Dict[str, Any]:

        return {
            "fatty_liver": self.fatty_agent,
            "fibrosis": self.fibrosis_agent,
            "cirrhosis": self.cirrhosis_agent,
            "tumor_classification": self.tumor_agent,
            "liver_segmentation": (
                self.segmentation_agent
            ),
            "clinical_reasoning": (
                self.clinical_reasoning_agent
            ),
        }

    def get_coordination_modules(
        self,
    ) -> Dict[str, Any]:

        return {
            "trust_manager": self.trust_manager,
            "adaptive_fusion": self.adaptive_fusion,
            "conflict_detector": self.conflict_detector,
            "decision_engine": self.decision_engine,
        }


# ============================================================
# END OF FILE
# ============================================================
