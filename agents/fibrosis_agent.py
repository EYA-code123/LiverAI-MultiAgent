```python
import numpy as np
import pandas as pd


class FibrosisAgent:

    def __init__(self, model):

        self.name = "FibrosisAgent"
        self.model_name = "XGBoost"
        self.model = model

        # ==================================================
        # FEATURES EXPECTED BY THE TRAINED XGBOOST MODEL
        # ==================================================

        self.features = [
            "age",
            "male",
            "weight",
            "height",
            "bmi",
            "futime",
            "days",
            "test",
            "value"
        ]

    # ======================================================
    # PREDICTION
    # ======================================================

    def predict(self, patient_data):

        # ==================================================
        # CREATE DATAFRAME
        # ==================================================

        if isinstance(patient_data, dict):

            X = pd.DataFrame(
                [patient_data]
            )

        elif isinstance(
            patient_data,
            pd.DataFrame
        ):

            X = patient_data.copy()

        else:

            X = pd.DataFrame(
                [patient_data],
                columns=self.features
            )

        # ==================================================
        # CHECK / ADD MISSING FEATURES
        # ==================================================

        for feature in self.features:

            if feature not in X.columns:

                X[feature] = np.nan

        # ==================================================
        # KEEP ONLY FEATURES EXPECTED BY THE MODEL
        # ==================================================

        X = X[
            self.features
        ].copy()

        # ==================================================
        # PREDICTION
        # ==================================================

        prediction = self.model.predict(
            X
        )[0]

        # ==================================================
        # PROBABILITIES
        # ==================================================

        probability = None
        class_probabilities = None

        if hasattr(
            self.model,
            "predict_proba"
        ):

            probabilities = (
                self.model.predict_proba(X)[0]
            )

            probability = float(
                np.max(probabilities)
            )

            class_probabilities = {
                f"class_{int(cls)}":
                    float(prob)
                for cls, prob in zip(
                    self.model.classes_,
                    probabilities
                )
            }

        # ==================================================
        # RESULT
        # ==================================================

        return {

            "agent": self.name,

            "model": self.model_name,

            "prediction": int(
                prediction
            ),

            "probability": probability,

            "class_probabilities":
                class_probabilities,

            "status": "completed"

        }
```
