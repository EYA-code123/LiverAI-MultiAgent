class EvidenceReasoner:

    def __init__(
        self,
        minimum_confidence=0.55
    ):

        self.minimum_confidence = float(
            minimum_confidence
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

        for result in results:

            if result.get(
                "prediction"
            ) is None:

                continue

            agent_id = result.get(
                "agent_id",
                result.get(
                    "agent",
                    "unknown"
                )
            )

            prediction = result.get(
                "prediction"
            )

            node_id = (
                f"{agent_id}:{prediction}"
            )

            nodes.append({

                "id":
                    node_id,

                "agent":
                    agent_id,

                "task":
                    result.get(
                        "task_type",
                        "unknown"
                    ),

                "prediction":
                    prediction,

                "confidence":
                    float(
                        result.get(
                            "confidence",
                            0.0
                        )
                    ),

                "trust":
                    float(
                        result.get(
                            "trust",
                            0.0
                        )
                    ),

                "quality":
                    float(
                        result.get(
                            "quality",
                            0.0
                        )
                    )
            })

        # -----------------------------------------------------
        # AGREEMENT / SUPPORT EDGES
        # -----------------------------------------------------

        for i in range(
            len(nodes)
        ):

            for j in range(
                i + 1,
                len(nodes)
            ):

                a = nodes[i]
                b = nodes[j]

                if (
                    a["prediction"]
                    ==
                    b["prediction"]
                ):

                    relation = "supports"

                elif (
                    a["task"]
                    ==
                    b["task"]
                ):

                    relation = "conflicts"

                else:

                    relation = "complements"

                edges.append({

                    "source":
                        a["id"],

                    "target":
                        b["id"],

                    "relation":
                        relation
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

        valid = [

            r

            for r in results

            if r.get(
                "prediction"
            ) is not None
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

            return (

                float(
                    result.get(
                        "trust",
                        0.0
                    )
                )

                *

                float(
                    result.get(
                        "confidence",
                        0.0
                    )
                )

                *

                float(
                    result.get(
                        "quality",
                        0.0
                    )
                )
            )

        best = max(
            valid,
            key=evidence_score
        )

        prediction = best.get(
            "prediction"
        )

        confidence = float(
            best.get(
                "confidence",
                0.0
            )
        )

        # -----------------------------------------------------
        # CONFLICT RESOLUTION CAN OVERRIDE
        # -----------------------------------------------------

        if conflict_resolution:

            resolved_prediction = (
                conflict_resolution.get(
                    "prediction"
                )
            )

            if resolved_prediction is not None:

                prediction = (
                    resolved_prediction
                )

        # -----------------------------------------------------
        # EXPLANATION
        # -----------------------------------------------------

        supporting_agents = [

            r.get(
                "agent_id",
                r.get(
                    "agent"
                )
            )

            for r in valid

            if r.get(
                "prediction"
            ) == prediction
        ]

        if supporting_agents:

            explanation = (

                f"The synthesized finding "
                f"'{prediction}' is supported "
                f"by: "
                f"{', '.join(supporting_agents)}."
            )

        else:

            explanation = (

                f"The finding '{prediction}' "
                f"is based on the strongest "
                f"available evidence."
            )

        return {

            "status":
                "completed",

            "prediction":
                prediction,

            "confidence":
                confidence,

            "uncertainty":
                1.0 - confidence,

            "supporting_agents":
                supporting_agents,

            "explanation":
                explanation,

            "evidence_graph":
                graph
        }
