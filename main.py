import json

from coordinator import (
    LiverAICoordinator
)


def build_coordinator():

    coordinator = (
        LiverAICoordinator()
    )

    # =========================================================
    # IMPORTANT
    # =========================================================
    #
    # Ajouter ici UNIQUEMENT les agents dont les modèles
    # existent réellement.
    #
    # Exemple :
    #
    # from agents.cirrhosis_agent import CirrhosisAgent
    #
    # coordinator.register_agent(
    #     agent_id="cirrhosis",
    #     agent=CirrhosisAgent(...),
    #     task_type="cirrhosis_classification",
    #     modality="clinical"
    # )

    return coordinator


def main():

    coordinator = (
        build_coordinator()
    )

    print(
        "=" * 70
    )

    print(
        "LiverAI - Adaptive Coordination Intelligence"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Registered agents:"
    )

    for agent_id in (
        coordinator.agents
    ):

        print(
            f"  - {agent_id}"
        )

    print()

    if not coordinator.agents:

        print(
            "No real agents are registered yet."
        )

        print(
            "Register only agents backed by real trained models."
        )

        return

    # ---------------------------------------------------------
    # Example input
    # ---------------------------------------------------------

    inputs = {}

    result = coordinator.run(

        patient_id=
            "P001",

        inputs=
            inputs
    )

    print(
        json.dumps(
            result,
            indent=2,
            default=str
        )
    )


if __name__ == "__main__":

    main()
