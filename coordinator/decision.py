class DecisionEngine:

    def decide(
        self,
        agent_results,
        conflicts,
        fused_results
    ):

        successful_agents = [
            r for r in agent_results
            if r.error is None
        ]

        decision = {
            "status": "completed",
            "num_agents": len(agent_results),
            "successful_agents": len(successful_agents),
            "failed_agents": (
                len(agent_results)
                - len(successful_agents)
            ),
            "conflicts": conflicts,
            "fused_results": fused_results
        }

        if conflicts:

            decision["warning"] = (
                "Conflicting agent predictions detected."
            )

        else:

            decision["warning"] = None

        return decision 
