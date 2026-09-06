# ============================================================
# LIVER AI — AGENTS PACKAGE
# Lazy imports to avoid loading optional dependencies
# ============================================================

__all__ = [
    "BaseAgent",
    "CirrhosisAgent",
    "FattyLiverAgent",
    "ClinicalReasoningAgent",
    "FibrosisAgent",
    "TumorClassificationAgent",
    "LiverSegmentationAgent",
]


def __getattr__(name):

    if name == "BaseAgent":
        from .base_agent import BaseAgent
        return BaseAgent

    if name == "CirrhosisAgent":
        from .cirrhosis_agent import CirrhosisAgent
        return CirrhosisAgent

    if name == "FattyLiverAgent":
        from .fatty_liver_agent import FattyLiverAgent
        return FattyLiverAgent

    if name == "ClinicalReasoningAgent":
        from .clinical_reasoning_agent import ClinicalReasoningAgent
        return ClinicalReasoningAgent

    if name == "FibrosisAgent":
        from .fibrosis_agent import FibrosisAgent
        return FibrosisAgent

    if name == "TumorClassificationAgent":
        from .tumor_classification_agent import TumorClassificationAgent
        return TumorClassificationAgent

    if name == "LiverSegmentationAgent":
        from .liver_segmentation_agent import LiverSegmentationAgent
        return LiverSegmentationAgent

    raise AttributeError(
        f"module 'agents' has no attribute '{name}'"
    )
