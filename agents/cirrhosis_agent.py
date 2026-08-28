import numpy as np
import pandas as pd

class CirrhosisAgent:

    def __init__(self, model_package):

        self.name = "CirrhosisAgent"
        self.model_name = "XGBoost"

        self.model = model_package["model"]

        self.feature_names = list(
            model_package["feature_names"]
        )

        self.numerical_columns = list(
            model_package["numerical_columns"]
        )

        self.categorical_columns = list(
            model_package["categorical_columns"]
        )

        self.encoders = model_package.get(
            "encoders",
            {}
        )

        self.target_encoder = model_package.get(
            "target_encoder",
            None
        )

        self.numerical_imputer = model_package[
            "numerical_imputer"
        ]

        self.categorical_imputer = model_package[
            "categorical_imputer"
        ]

        self.model_features = list(
            self.feature_names
        )

        if hasattr(
            self.numerical_imputer,
            "feature_names_in_"
        ):

            self.imputer_numerical_features = list(
                self.numerical_imputer.feature_names_in_
            )

        else:

            self.imputer_numerical_features = list(
                self.numerical_columns
            )

        if hasattr(
            self.categorical_imputer,
            "feature_names_in_"
        ):

            self.imputer_categorical_features = list(
                self.categorical_imputer.feature_names_in_
            )

        else:

            self.imputer_categorical_features = list(
                self.categorical_columns
            )

    def predict(self, patient_data):

        print("=" * 70)
        print("CIRRHOSIS AGENT")
        print("=" * 70)

        # ==================================================
        # CREATE DATAFRAME
        # ==================================================

        if isinstance(
            patient_data,
            dict
        ):

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
                columns=self.feature_names
            )

        # ==================================================
        # ADD MISSING MODEL FEATURES
        # ==================================================

        for col in self.feature_names:

            if col not in X.columns:

                X[col] = np.nan

        # ==================================================
        # TEMPORARY FEATURES FOR IMPUTER
        # ==================================================

        for col in self.imputer_numerical_features:

            if col not in X.columns:

                index = (
                    self.imputer_numerical_features.index(col)
                )

                if hasattr(
                    self.numerical_imputer,
                    "statistics_"
                ):

                    value = (
                        self.numerical_imputer.statistics_[index]
                    )

                else:

                    value = np.nan

                X[col] = value

        # ==================================================
        # NUMERICAL IMPUTATION
        # ==================================================

        if len(
            self.imputer_numerical_features
        ) > 0:

            numerical_data = X[
                self.imputer_numerical_features
            ].copy()

            numerical_data = (
                self.numerical_imputer.transform(
                    numerical_data
                )
            )

            numerical_data = pd.DataFrame(
                numerical_data,
                columns=self.imputer_numerical_features,
                index=X.index
            )

            X[
                self.imputer_numerical_features
            ] = numerical_data

        # ==================================================
        # REMOVE STAGE
        # ==================================================

        if "Stage" in X.columns:

            X = X.drop(
                columns=["Stage"]
            )

        # ==================================================
        # CATEGORICAL IMPUTATION
        # ==================================================

        categorical_features = [
            col
            for col in self.categorical_columns
            if col in self.feature_names
        ]

        if len(
            categorical_features
        ) > 0:

            categorical_data = X[
                categorical_features
            ].copy()

            categorical_data = (
                self.categorical_imputer.transform(
                    categorical_data
                )
            )

            categorical_data = pd.DataFrame(
                categorical_data,
                columns=categorical_features,
                index=X.index
            )

            X[
                categorical_features
            ] = categorical_data

        # ==================================================
        # ENCODING
        # ==================================================

        for col in categorical_features:

            if col in self.encoders:

                encoder = self.encoders[col]

                values = X[col].astype(str)

                known_values = set(
                    encoder.classes_
                )

                values = values.apply(
                    lambda value:
                    value
                    if value in known_values
                    else encoder.classes_[0]
                )

                X[col] = encoder.transform(
                    values
                )

        # ==================================================
        # FINAL FEATURES
        # ==================================================

        X = X[
            self.model_features
        ].copy()

        # ==================================================
        # PREDICTION
        # ==================================================

        prediction_encoded = (
            self.model.predict(X)[0]
        )

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
                np.max(probabilities)
            )

        # ==================================================
        # DECODE TARGET
        # ==================================================

        prediction = prediction_encoded

        if self.target_encoder is not None:

            try:

                prediction = (
                    self.target_encoder.inverse_transform(
                        [prediction_encoded]
                    )[0]
                )

            except Exception:

                prediction = prediction_encoded

        # ==================================================
        # RESULT
        # ==================================================

       return {
    "agent": self.name,
    "model": self.model_name,
    "status": "completed",
    "prediction": str(prediction),
    "probability": probability
}
