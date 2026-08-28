import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image


class TumorClassificationAgent:

    def __init__(self, model_path):

        self.name = "Tumor Classification Agent"
        self.model_name = "EfficientNet / MobileNet"

        self.model_path = model_path

        self.classes = [
            "Angiosarcoma",
            "Cholangiocarcinoma",
            "Healthy",
            "Hemangioma",
            "Hepatocellular Carcinoma"
        ]

        self.model = load_model(
            model_path,
            compile=False
        )

    def preprocess(self, image):

        if isinstance(image, str):

            image = Image.open(
                image
            ).convert("RGB")

        elif isinstance(image, np.ndarray):

            image = Image.fromarray(
                image.astype(np.uint8)
            ).convert("RGB")

        elif not isinstance(image, Image.Image):

            raise TypeError(
                "Image must be a path, numpy array or PIL Image"
            )

        image = image.resize(
            (224, 224)
        )

        image = np.asarray(
            image,
            dtype=np.float32
        )

        image /= 255.0

        image = np.expand_dims(
            image,
            axis=0
        )

        return image

    def predict(self, image):

        try:

            if image is None:
                raise ValueError(
                    "MRI image is None"
                )

            x = self.preprocess(image)

            predictions = self.model.predict(
                x,
                verbose=0
            )

            probabilities = predictions[0]

            predicted_index = int(
                np.argmax(probabilities)
            )

            predicted_class = self.classes[
                predicted_index
            ]

            confidence = float(
                probabilities[predicted_index]
            )

            class_probabilities = {
                self.classes[i]:
                    float(probabilities[i])
                for i in range(
                    len(self.classes)
                )
            }

            return {
                "agent": self.name,
                "model": self.model_name,
                "status": "completed",
                "prediction": predicted_class,
                "probability": confidence,
                "class_probabilities":
                    class_probabilities
            }

        except Exception as e:

            return {
                "agent": self.name,
                "model": self.model_name,
                "status": "error",
                "prediction": None,
                "probability": None,
                "error": str(e)
            }
