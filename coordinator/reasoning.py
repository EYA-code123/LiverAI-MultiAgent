"""
Reasoning Intelligence
=======================

Cross-task evidence synthesis.

This module does not pretend that different medical tasks
share the same label space.

Instead it creates a structured evidence summary.
"""


class EvidenceReasoner:

    def __init__(self):

        self.rules = {
            "cirrhosis": self._cirrhosis_rule,
            "fibrosis": self._fibrosis_rule,
            "fatty_liver": self._fatty_liver_rule,
            "tumor": self._tumor_rule,
            "tumor_classification":
                self._tumor_rule,
            "segmentation":
                self._segmentation_rule,
        }

    def _cirrhosis_rule(
        self,
        result
    ):

        prediction = result.get(
            "prediction"
        )

        confidence = result.get(
            "confidence",
            0.0
        )

        return {
            "finding":
                f"Cirrhosis prediction: {prediction}",
            "confidence":
                confidence
        }

    def _fibrosis_rule(
        self,
        result
    ):

        return {
            "finding":
                f"Fibrosis prediction: "
                f"{result.get('prediction')}",
            "confidence":
                result.get("confidence", 0.0)
        }

    def _fatty_liver_rule(
        self,
        result
    ):

        return {
            "finding":
                f"Fatty liver prediction: "
                f"{result.get('prediction')}",
            "confidence":
                result.get("confidence", 0.0)
        }

    def _tumor_rule(
        self,
        result
    ):

        return {
            "finding":
                f"Tumor classification: "
                f"{result.get('prediction')}",
            "confidence":
                result.get("confidence", 0.0)
        }

    def _segmentation_rule(
        self,
        result
    ):

        return {
            "finding":
                "Liver segmentation available.",
            "confidence":
                result.get(
                    "segmentation_quality",
                    result.get(
                        "confidence",
                        0.0
                    )
                )
        }

    def synthesize(
        self,
        fused_results,
        conflicts=None
    ):

        conflicts = conflicts or []

        evidence = []

        for task_type, result in fused_results.items():

            if result is None:
                continue

            rule = self.rules.get(
                task_type
            )

            if rule:

                item = rule(result)

            else:

                item = {
                    "finding":
                        f"{task_type}: "
                        f"{result.get('prediction')}",
                    "confidence":
                        result.get(
                            "confidence",
                            0.0
                        )
                }

            item["task_type"] = task_type

            evidence.append(item)

        conflict_count = len(
            conflicts
        )

        if conflict_count > 0:

            conflict_statement = (
                f"{conflict_count} intra-task "
                "prediction conflict(s) detected."
            )

        else:

            conflict_statement = (
                "No direct intra-task conflicts detected."
            )

        overall_confidence = (
            sum(
                item["confidence"]
                for item in evidence
            )
            / len(evidence)
            if evidence
            else 0.0
        )

        return {
            "evidence": evidence,
            "conflict_summary":
                conflict_statement,
            "overall_confidence":
                overall_confidence,
            "reasoning": self._build_reasoning(
                evidence,
                conflicts
            )
        }

    def _build_reasoning(
        self,
        evidence,
        conflicts
    ):

        if not evidence:

            return (
                "Insufficient evidence for "
                "clinical synthesis."
            )

        statements = [
            item["finding"]
            for item in evidence
        ]

        reasoning = (
            "Evidence synthesis: "
            + "; ".join(statements)
            + "."
        )

        if conflicts:

            reasoning += (
                " Some specialist predictions "
                "are discordant; additional "
                "clinical verification may be "
                "appropriate."
            )

        return reasoning
