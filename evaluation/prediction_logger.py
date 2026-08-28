import pandas as pd
from pathlib import Path


class PredictionLogger:

    def __init__(self, output_dir="results/predictions"):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def save(
        self,
        agent_id,
        patient_ids,
        y_true,
        y_pred,
        probabilities
    ):

        data = []

        for i in range(len(patient_ids)):

            probability = probabilities[i]

            if hasattr(probability, "__len__"):
                confidence = max(probability)
            else:
                confidence = max(
                    probability,
                    1 - probability
                )

            data.append({
                "patient_id": patient_ids[i],
                "true_label": y_true[i],
                "predicted_label": y_pred[i],
                "probability": probability,
                "confidence": confidence,
                "correct": int(
                    y_true[i] == y_pred[i]
                )
            })

        df = pd.DataFrame(data)

        output_file = (
            self.output_dir /
            f"{agent_id}_predictions.csv"
        )

        df.to_csv(
            output_file,
            index=False
        )

        print(
            f"[SAVED] {output_file}"
        )

        return df
