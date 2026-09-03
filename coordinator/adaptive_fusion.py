from collections import defaultdict


class AdaptiveFusion:

    def __init__(
        self,
        minimum_weight=0.001
    ):

        self.minimum_weight = float(
            minimum_weight
        )

    # =========================================================
    # COMPUTE ADAPTIVE WEIGHT
    # =========================================================

    def compute_weight(
        self,
        result
    ):

        trust = float(
            result.get(
                "trust",
                0.0
            )
        )

        confidence = float(
            result.get(
                "confidence",
                0.0
            )
        )

        quality = float(
            result.get(
                "quality",
                0.0
            )
        )

        uncertainty = float(
            result.get(
                "uncertainty",
                1.0
            )
        )

        agreement = float(
            result.get(
                "agreement",
                0.5
            )
        )

        stability = float(
            result.get(
                "stability",
                0.5
            )
        )

        utility = float(
            result.get(
                "utility",
                0.5
            )
        )

        raw = (

            0.35 * trust

            + 0.20 * confidence

            + 0.15 * quality

            + 0.10 * (
                1.0 - uncertainty
            )

            + 0.10 * agreement

            + 0.05 * stability

            + 0.05 * utility
        )

        return max(
            self.minimum_weight,
            raw
        )

    # =========================================================
    # MAIN FUSION
    # =========================================================

    def fuse(
        self,
        results
    ):

        valid = [

            result

            for result in results

            if isinstance(
                result,
                dict
            )

            and result.get(
                "status",
                "success"
            )
            in (
                "success",
                "completed"
            )

            and result.get(
                "prediction"
            ) is not None
        ]

        if not valid:

            return {

                "status":
                    "no_valid_results",

                "evidence": [],

                "weights": {},

                "task_groups": {},

                "same_task_fusion": {}
            }

        # -----------------------------------------------------
        # WEIGHTS
        # -----------------------------------------------------

        raw_weights = {}

        for result in valid:

            agent_id = str(
                result.get(
                    "agent_id",
                    result.get(
                        "agent",
                        "unknown"
                    )
                )
            )

            raw_weights[
                agent_id
            ] = self.compute_weight(
                result
            )

        total_weight = sum(
            raw_weights.values()
        )

        weights = {

            agent_id:
                weight / total_weight

            for agent_id, weight
            in raw_weights.items()
        }

        # -----------------------------------------------------
        # EVIDENCE
        # -----------------------------------------------------

        evidence = []

        for result in valid:

            agent_id = str(
                result.get(
                    "agent_id",
                    result.get(
                        "agent",
                        "unknown"
                    )
                )
            )

            evidence.append({

                "agent_id":
                    agent_id,

                "task_type":
                    result.get(
                        "task_type",
                        "unknown"
                    ),

                "modality":
                    result.get(
                        "modality",
                        "unknown"
                    ),

                "prediction":
                    result.get(
                        "prediction"
                    ),

                "probability":
                    result.get(
                        "probability"
                    ),

                "class_probabilities":
                    result.get(
                        "class_probabilities",
                        {}
                    ),

                "confidence":
                    result.get(
                        "confidence",
                        0.0
                    ),

                "uncertainty":
                    result.get(
                        "uncertainty",
                        1.0
                    ),

                "quality":
                    result.get(
                        "quality",
                        0.0
                    ),

                "trust":
                    result.get(
                        "trust",
                        0.0
                    ),

                "agreement":
                    result.get(
                        "agreement",
                        0.5
                    ),

                "stability":
                    result.get(
                        "stability",
                        0.5
                    ),

                "utility":
                    result.get(
                        "utility",
                        0.5
                    ),

                "adaptive_weight":
                    weights.get(
                        agent_id,
                        0.0
                    ),

                "explanation":
                    result.get(
                        "explanation"
                    )
            })

        # -----------------------------------------------------
        # GROUP BY TASK
        # -----------------------------------------------------

        task_groups = (
            defaultdict(list)
        )

        for item in evidence:

            task_groups[
                item["task_type"]
            ].append(
                item
            )

        # -----------------------------------------------------
        # SAME TASK FUSION
        # -----------------------------------------------------

        same_task_fusion = {}

        for task, items in (
            task_groups.items()
        ):

            if len(items) >= 2:

                fused = (
                    self.fuse_same_task(
                        items
                    )
                )

                if fused is not None:

                    same_task_fusion[
                        task
                    ] = fused

        return {

            "status":
                "success",

            "evidence":
                evidence,

            "weights":
                weights,

            "task_groups":
                dict(task_groups),

            "same_task_fusion":
                same_task_fusion
        }

    # =========================================================
    # SAME TASK PROBABILITY FUSION
    # =========================================================

    def fuse_same_task(
        self,
        items
    ):

        probability_maps = []

        for item in items:

            probabilities = (
                item.get(
                    "class_probabilities",
                    {}
                )
            )

            if not isinstance(
                probabilities,
                dict
            ):

                continue

            if not probabilities:

                continue

            weight = float(
                item.get(
                    "adaptive_weight",
                    0.0
                )
            )

            probability_maps.append(
                (
                    probabilities,
                    weight
                )
            )

        if not probability_maps:

            return None

        classes = set()

        for probabilities, _ in (
            probability_maps
        ):

            classes.update(
                probabilities.keys()
            )

        fused = {
            cls: 0.0
            for cls in classes
        }

        total_weight = sum(

            weight

            for _, weight
            in probability_maps
        )

        if total_weight <= 0:

            return None

        for probabilities, weight in (
            probability_maps
        ):

            for cls in classes:

                fused[cls] += (

                    weight
                    * float(
                        probabilities.get(
                            cls,
                            0.0
                        )
                    )
                )

        for cls in fused:

            fused[cls] /= total_weight

        prediction = max(
            fused,
            key=fused.get
        )

        confidence = float(
            fused[prediction]
        )

        return {

            "prediction":
                prediction,

            "class_probabilities":
                fused,

            "confidence":
                confidence,

            "uncertainty":
                1.0 - confidence,

            "num_agents":
                len(
                    probability_maps
                )
        }
