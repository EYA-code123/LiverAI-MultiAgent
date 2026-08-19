%cd /content/LiverAI-MultiAgent

from pathlib import Path

path = Path("agents/fibrosis_agent.py")

code = '''import pandas as pd


class FibrosisAgent:

    def __init__(self, model):

        self.name = "FibrosisAgent"
        self.model_name = "XGBoost"
        self.model = model

        self.features = [
            "age",
            "male",
            "weight",
            "height",
            "bmi",
            "futime"
        ]

    def predict(self, patient_data):

        # ==================================================
        # CREATE DATAFRAME
        # ==================================================

        if isinstance(patient_data, dict):

            X = pd.DataFrame(
                [patient_data]
            )

        elif isinstance(patient_data, pd.DataFrame):

            X = patient_data.copy()

        else:

            X = pd.DataFrame(
                [patient_data],
                columns=self.features
            )

        # ==================================================
        # CHECK FEATURES
        # ==================================================

        missing_features = [
            feature
            for feature in self.features
            if feature not in X.columns
        ]

        if missing_features:

            raise ValueError(
                f"Missing features: {missing_features}"
            )

        # ==================================================
        # EXACT FEATURE ORDER
        # ==================================================

        X = X[
            self.features
        ].copy()

        # ==================================================
        # PREDICTION
        # ==================================================

        prediction = self.model.predict(X)[0]

        # ==================================================
        # PROBABILITY
        # ==================================================

        probability = None

        if hasattr(
            self.model,
            "predict_proba"
        ):

            probabilities = (
                self.model.predict_proba(X)[0]
            )

            probability = float(
                max(probabilities)
            )

        # ==================================================
        # RESULT
        # ==================================================

        return {

            "agent": self.name,

            "model": self.model_name,

            "prediction": str(
                prediction
            ),

            "probability": probability,

            "status": "completed"
        }
'''

path.write_text(
    code,
    encoding="utf-8"
)

print("✅ File physically replaced:")
print(path)

print("\nFile content:")
print(path.read_text(encoding="utf-8"))
