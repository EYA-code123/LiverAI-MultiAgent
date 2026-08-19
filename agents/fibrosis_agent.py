%cd /content/LiverAI-MultiAgent

agent_code = '''
import pandas as pd


class FibrosisAgent:

    def __init__(self, model):

        self.name = "FibrosisAgent"
        self.model_name = "XGBoost"
        self.model = model

        # Exact features used during training
        self.features = [
            "age",
            "male",
            "weight",
            "height",
            "bmi",
            "futime"
        ]

    def predict(self, patient_data):

        # Create DataFrame
        if isinstance(patient_data, dict):

            X = pd.DataFrame([patient_data])

        elif isinstance(patient_data, pd.DataFrame):

            X = patient_data.copy()

        else:

            X = pd.DataFrame(
                [patient_data],
                columns=self.features
            )

        # Check required features
        missing_features = [
            feature
            for feature in self.features
            if feature not in X.columns
        ]

        if missing_features:

            raise ValueError(
                f"Missing features: {missing_features}"
            )

        # Keep exact feature order
        X = X[self.features].copy()

        # Prediction
        prediction = self.model.predict(X)[0]

        # Probability
        probability = None

        if hasattr(self.model, "predict_proba"):

            probabilities = self.model.predict_proba(X)[0]

            probability = float(
                max(probabilities)
            )

        return {

            "agent": self.name,

            "model": self.model_name,

            "prediction": str(prediction),

            "probability": probability,

            "status": "completed"
        }
'''

with open(
    "agents/fibrosis_agent.py",
    "w",
    encoding="utf-8"
) as f:

    f.write(agent_code)

print("✅ fibrosis_agent.py updated")
