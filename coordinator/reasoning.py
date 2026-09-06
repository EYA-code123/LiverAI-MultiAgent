class EvidenceReasoner:

    def __init__(
        self,
        minimum_confidence=0.55
    ):

        self.minimum_confidence = float(
            minimum_confidence
        )

    # =========================================================
    # CHECK VALID EVIDENCE
    # =========================================================

    def _is_valid_evidence(
        self,
        result
    ):

        task_type = result.get(
            "task_type",
            "unknown"
        )

        status = result.get(
            "status",
            "success"
        )

        if status not in (
            "success",
            "completed"
        ):
            return False

        # -----------------------------------------------------
        # LIVER SEGMENTATION
        # -----------------------------------------------------
        # Segmentation does not require a class prediction.
        # Its valid evidence is the liver mask and/or
        # probability map.
        # -----------------------------------------------------

        if task_type == "liver_segmentation":

            details = result.get(
                "details",
                {}
            )

            return (
                isinstance(
                    details,
                    dict
                )
                and (
                    "liver_mask" in details
                    or "probability_map" in details
                )
            )

        # -----------------------------------------------------
        # OTHER TASKS
        # -----------------------------------------------------

        return (
            result.get(
                "prediction"
            ) is not None
        )

    # =========================================================
    # BUILD EVIDENCE GRAPH
    # =========================================================

    def build_evidence_graph(
        self,
        results
    ):

        nodes = []
        edges = []

        valid_results = []

        for result in results:

            if not self._is_valid_evidence(
                result
            ):
                continue

            valid_results.append(
                result
            )

        # -----------------------------------------------------
        # BUILD NODES
        # -----------------------------------------------------

        for index, result in enumerate(
            valid_results
        ):

            agent_id = result.get(
                "agent_id",
                result.get(
                    "agent",
                    "unknown"
                )
            )

            task_type = result.get(
                "task_type",
                "unknown"
            )

            prediction = result.get(
                "prediction"
            )

            confidence = float(
                result.get(
                    "confidence",
                    0.0
                )
            )

            trust = float(
                result.get(
                    "trust",
                    0.0
                )
            )

            quality = float(
                result.get(
                    "quality",
                    0.0
                )
            )

            # -------------------------------------------------
            # NODE ID
            # -------------------------------------------------
            # Segmentation has no prediction, therefore its
            # node ID is based on the task instead.
            # -------------------------------------------------

            if (
                task_type
                == "liver_segmentation"
            ):

                node_id = (
                    f"{agent_id}:"
                    f"{task_type}"
                )

            else:

                node_id = (
                    f"{agent_id}:"
                    f"{prediction}"
                )

            nodes.append({

                "id":
                    node_id,

                "agent":
                    agent_id,

                "task":
                    task_type,

                "prediction":
                    prediction,

                "confidence":
                    confidence,

                "trust":
                    trust,

                "quality":
                    quality
            })

        # -----------------------------------------------------
        # BUILD EDGES
        # -----------------------------------------------------

        for i in range(
            len(nodes)
        ):

            for j in range(
                i + 1,
                len(nodes)
            ):

                node_a = nodes[i]
                node_b = nodes[j]

                task_a = node_a.get(
                    "task"
                )

                task_b = node_b.get(
                    "task"
                )

                # -------------------------------------------------
                # SAME TASK
                # -------------------------------------------------

                if task_a == task_b:

                    prediction_a = node_a.get(
                        "prediction"
                    )

                    prediction_b = node_b.get(
                        "prediction"
                    )

                    if (
                        prediction_a
                        == prediction_b
                    ):

                        relation = (
                            "supports"
                        )

                    else:

                        relation = (
                            "conflicts"
                        )

                # -------------------------------------------------
                # DIFFERENT TASKS
                # -------------------------------------------------

                else:

                    relation = (
                        "complements"
                    )

                edges.append({

                    "source":
                        node_a["id"],

                    "target":
                        node_b["id"],

                    "relation":
                        relation,

                    "task_type":
                        None
                })

        return {

            "nodes":
                nodes,

            "edges":
                edges
        }

    # =========================================================
    # SYNTHESIS
    # =========================================================

    def synthesize(
        self,
        results,
        conflict_resolution=None
    ):

        graph = (
            self.build_evidence_graph(
                results
            )
        )

        # -----------------------------------------------------
        # VALID EVIDENCE
        # -----------------------------------------------------

        valid = [

            r

            for r in results

            if self._is_valid_evidence(
                r
            )
        ]

        if not valid:

            return {

                "status":
                    "insufficient_evidence",

                "prediction":
                    None,

                "confidence":
                    0.0,

                "explanation":
                    "No valid agent evidence.",

                "evidence_graph":
                    graph
            }

        # -----------------------------------------------------
        # BEST EVIDENCE
        # -----------------------------------------------------

        def evidence_score(
            result
        ):

            confidence = float(
                result.get(
                    "confidence",
                    0.0
                )
            )

            trust = float(
                result.get(
                    "trust",
                    0.0
                )
            )

            quality = float(
                result.get(
                    "quality",
                    0.0
                )
            )

            return (
                confidence
                * trust
                * quality
            )

        best = max(
            valid,
            key=evidence_score
        )

        # -----------------------------------------------------
        # TASK-SPECIFIC EXPLANATION
        # -----------------------------------------------------

        task_types = list({

            r.get(
                "task_type",
                "unknown"
            )

            for r in valid

        })

        # -----------------------------------------------------
        # SEGMENTATION-ONLY CASE
        # -----------------------------------------------------

        if (
            len(task_types) == 1
            and task_types[0]
            == "liver_segmentation"
        ):

            return {

                "status":
                    "completed",

                "prediction":
                    None,

                "confidence":
                    float(
                        best.get(
                            "confidence",
                            0.0
                        )
                    ),

                "explanation":
                    (
                        "Valid liver segmentation "
                        "evidence was detected. "
                        "The segmentation output is "
                        "represented by the liver mask "
                        "and probability map rather "
                        "than a class prediction."
                    ),

                "evidence_graph":
                    graph
            }

        # -----------------------------------------------------
        # GENERAL CASE
        # -----------------------------------------------------

        prediction = best.get(
            "prediction"
        )

        confidence = float(
            best.get(
                "confidence",
                0.0
            )
        )

        if conflict_resolution:

            explanation = (
                "Evidence was evaluated "
                "with task-specific conflict "
                "resolution."
            )

        else:

            explanation = (
                "Best available evidence "
                "selected using confidence, "
                "trust and quality."
            )

        return {

            "status":
                "completed",

            "prediction":
                prediction,

            "confidence":
                confidence,

            "explanation":
                explanation,

            "evidence_graph":
                graph
        }

    # =========================================================
    # TASK ASSESSMENT
    # =========================================================

    def assess_task(
        self,
        task_type,
        results,
        conflict_resolver=None
    ):

        task_results = [

            r

            for r in results

            if r.get(
                "task_type"
            ) == task_type
        ]

        if not task_results:

            return {

                "prediction":
                    None,

                "confidence":
                    0.0,

                "resolution": {

                    "status":
                        "unresolved",

                    "consensus":
                        False,

                    "prediction":
                        None,

                    "consensus_strength":
                        0.0,

                    "reason":
                        "No results for this task."
                }
            }

        valid_results = [

            r

            for r in task_results

            if self._is_valid_evidence(
                r
            )
        ]

        if not valid_results:

            return {

                "prediction":
                    None,

                "confidence":
                    0.0,

                "resolution": {

                    "status":
                        "unresolved",

                    "consensus":
                        False,

                    "prediction":
                        None,

                    "consensus_strength":
                        0.0,

                    "reason":
                        "No valid evidence."
                }
            }

        # -----------------------------------------------------
        # SEGMENTATION
        # -----------------------------------------------------

        if (
            task_type
            == "liver_segmentation"
        ):

            best = max(

                valid_results,

                key=lambda r:
                    float(
                        r.get(
                            "confidence",
                            0.0
                        )
                    )
            )

            return {

                "prediction":
                    None,

                "confidence":
                    float(
                        best.get(
                            "confidence",
                            0.0
                        )
                    ),

                "resolution": {

                    "status":
                        "resolved",

                    "consensus":
                        True,

                    "prediction":
                        None,

                    "consensus_strength":
                        float(
                            best.get(
                                "confidence",
                                0.0
                            )
                        ),

                    "scores":
                        {},

                    "reason":
                        (
                            "Valid liver segmentation "
                            "evidence detected."
                        )
                }
            }

        # -----------------------------------------------------
        # NORMAL CLASSIFICATION TASK
        # -----------------------------------------------------

        if conflict_resolver:

            resolution = (
                conflict_resolver.resolve(
                    task_type,
                    task_results
                )
            )

        else:

            best = max(

                valid_results,

                key=lambda r:
                    float(
                        r.get(
                            "confidence",
                            0.0
                        )
                    )
            )

            resolution = {

                "status":
                    "resolved",

                "consensus":
                    True,

                "prediction":
                    best.get(
                        "prediction"
                    ),

                "consensus_strength":
                    float(
                        best.get(
                            "confidence",
                            0.0
                        )
                    ),

                "reason":
                    "Best valid evidence selected."
            }

        # -----------------------------------------------------
        # CONFIDENCE
        # -----------------------------------------------------

        best_prediction = resolution.get(
            "prediction"
        )

        matching = [

            r

            for r in valid_results

            if r.get(
                "prediction"
            ) == best_prediction
        ]

        if matching:

            confidence = max(

                float(
                    r.get(
                        "confidence",
                        0.0
                    )
                )

                for r in matching
            )

        else:

            confidence = 0.0

        return {

            "prediction":
                best_prediction,

            "confidence":
                confidence,

            "resolution":
                resolution
        }
