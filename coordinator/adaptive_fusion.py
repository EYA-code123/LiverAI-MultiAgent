def _run_adaptive_fusion(self, results):
    """
    Run Adaptive Fusion on normalized agent results.

    Parameters
    ----------
    results : list
        List of agent result dictionaries.

    Returns
    -------
    dict
        Fusion result.
    """

    # ---------------------------------------------------------
    # No results
    # ---------------------------------------------------------
    if not results:
        return {
            "status": "unavailable",
            "reason": "No agent results available",
            "results": []
        }

    # ---------------------------------------------------------
    # Adaptive Fusion not initialized
    # ---------------------------------------------------------
    if self.adaptive_fusion is None:
        return {
            "status": "unavailable",
            "reason": "AdaptiveFusion is not initialized",
            "results": results
        }

    # ---------------------------------------------------------
    # Normalize input
    # ---------------------------------------------------------
    valid_results = []

    for result in results:

        if result is None:
            continue

        if isinstance(result, dict):
            valid_results.append(result)

        elif hasattr(result, "to_dict"):
            try:
                valid_results.append(result.to_dict())
            except Exception:
                continue

    if not valid_results:
        return {
            "status": "unavailable",
            "reason": "No valid results available for fusion",
            "results": []
        }

    # ---------------------------------------------------------
    # Run Adaptive Fusion
    # ---------------------------------------------------------
    try:

        fused_result = self.adaptive_fusion.fuse(
            valid_results
        )

        # -----------------------------------------------------
        # Fusion returned None
        # -----------------------------------------------------
        if fused_result is None:

            return {
                "status": "failed",
                "error": "AdaptiveFusion returned None",
                "results": valid_results
            }

        # -----------------------------------------------------
        # Fusion returned a dictionary
        # -----------------------------------------------------
        if isinstance(fused_result, dict):

            # Keep original results available
            if "results" not in fused_result:
                fused_result["results"] = valid_results

            return fused_result

        # -----------------------------------------------------
        # Unexpected return type
        # -----------------------------------------------------
        return {
            "status": "failed",
            "error": (
                "AdaptiveFusion returned an unsupported type: "
                f"{type(fused_result).__name__}"
            ),
            "results": valid_results
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
            "results": valid_results
        }
