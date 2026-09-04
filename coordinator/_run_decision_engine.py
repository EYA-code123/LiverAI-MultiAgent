def _run_decision_engine(
    self,
    results,
    conflicts,
    clinical_result
):
    """
    Run the Decision Engine.

    Parameters
    ----------
    results : list
        Normalized agent results.

    conflicts : list
        Conflicts detected between agents.

    clinical_result : dict
        Clinical reasoning result.

    Returns
    -------
    dict
        Final decision result.
    """

    # ---------------------------------------------------------
    # Decision Engine not initialized
    # ---------------------------------------------------------
    if self.decision_engine is None:

        return {
            "status": "unavailable",
            "reason": "DecisionEngine is not initialized"
        }

    # ---------------------------------------------------------
    # Normalize agent results
    # ---------------------------------------------------------
    valid_results = []

    if results is not None:

        for result in results:

            if result is None:
                continue

            if isinstance(result, dict):

                valid_results.append(result)

            elif hasattr(result, "to_dict"):

                try:
                    valid_results.append(
                        result.to_dict()
                    )
                except Exception:
                    continue

    # ---------------------------------------------------------
    # Normalize conflicts
    # ---------------------------------------------------------
    valid_conflicts = []

    if conflicts is not None:

        if isinstance(conflicts, list):

            valid_conflicts = conflicts

        else:

            valid_conflicts = [conflicts]

    # ---------------------------------------------------------
    # Normalize Clinical Reasoning result
    # ---------------------------------------------------------
    if isinstance(clinical_result, dict):

        clinical_dict = dict(clinical_result)

    elif clinical_result is not None and hasattr(
        clinical_result,
        "to_dict"
    ):

        try:

            clinical_dict = clinical_result.to_dict()

        except Exception:

            clinical_dict = {}

    else:

        clinical_dict = {}

    # ---------------------------------------------------------
    # Improve Clinical Reasoning confidence
    # ---------------------------------------------------------
    #
    # The Clinical Reasoning agent currently returns something
    # similar to:
    #
    # {
    #     "prediction": 1,
    #     "probabilities": {
    #         "selector_0_probability": 0.24,
    #         "selector_1_probability": 0.75
    #     }
    # }
    #
    # Therefore confidence should not remain 0.0.
    # ---------------------------------------------------------

    probabilities = clinical_dict.get(
        "probabilities"
    )

    if isinstance(probabilities, dict):

        numeric_probabilities = []

        for value in probabilities.values():

            try:

                value = float(value)

                if 0.0 <= value <= 1.0:
                    numeric_probabilities.append(value)

            except (
                TypeError,
                ValueError
            ):

                continue

        if numeric_probabilities:

            clinical_confidence = max(
                numeric_probabilities
            )

            clinical_uncertainty = (
                1.0 - clinical_confidence
            )

            clinical_dict["confidence"] = (
                clinical_confidence
            )

            clinical_dict["uncertainty"] = (
                clinical_uncertainty
            )

    # ---------------------------------------------------------
    # If confidence exists but uncertainty doesn't
    # ---------------------------------------------------------
    if (
        "confidence" in clinical_dict
        and "uncertainty" not in clinical_dict
    ):

        try:

            clinical_confidence = float(
                clinical_dict["confidence"]
            )

            clinical_dict["uncertainty"] = (
                1.0 - clinical_confidence
            )

        except (
            TypeError,
            ValueError
        ):

            pass

    # ---------------------------------------------------------
    # Run Decision Engine
    # ---------------------------------------------------------
    try:

        decision = self.decision_engine.decide(
            valid_results,
            valid_conflicts,
            clinical_dict
        )

        # -----------------------------------------------------
        # Decision Engine returned None
        # -----------------------------------------------------
        if decision is None:

            return {
                "status": "failed",
                "error": "DecisionEngine returned None",
                "results": valid_results,
                "conflicts": valid_conflicts,
                "clinical_reasoning": clinical_dict
            }

        # -----------------------------------------------------
        # Dictionary result
        # -----------------------------------------------------
        if isinstance(decision, dict):

            return decision

        # -----------------------------------------------------
        # Unexpected return type
        # -----------------------------------------------------
        return {
            "status": "failed",
            "error": (
                "DecisionEngine returned an unsupported type: "
                f"{type(decision).__name__}"
            ),
            "results": valid_results,
            "conflicts": valid_conflicts,
            "clinical_reasoning": clinical_dict
        }

    # ---------------------------------------------------------
    # Exception
    # ---------------------------------------------------------
    except Exception as exc:

        import traceback

        return {
            "status": "failed",
            "error": str(exc),
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
            "results": valid_results,
            "conflicts": valid_conflicts,
            "clinical_reasoning": clinical_dict
        }v 
