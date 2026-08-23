import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image


class TumorClassificationAgent:

    def __init__(self, model_path):
        self.name = "Tumor Classification Agent"

        self.model_path = model_path

        self.classes = [
            "Angiosarcoma",
            "Cholangiocarcinoma",
            "Healthy",
            "Hemangioma",
            "Hepatocellular Carcinoma"
        ]

        self.model = load_model(model_path)

    def preprocess(self, image):
        """
        Preprocess MRI image.
        """

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        image = image.resize((224, 224))

        image = np.array(image, dtype=np.float32)

        image = image / 255.0

        image = np.expand_dims(image, axis=0)

        return image

    def predict(self, image):
        """
        Perform tumor classification.
        """

        try:

            x = self.preprocess(image)

            predictions = self.model.predict(x, verbose=0)

            probabilities = predictions[0]

            predicted_index = int(np.argmax(probabilities))

            predicted_class = self.classes[predicted_index]

            confidence = float(probabilities[predicted_index])

            class_probabilities = {
                self.classes[i]: float(probabilities[i])
                for i in range(len(self.classes))
            }

            return {
                "agent": self.name,
                "status": "success",
                "prediction": predicted_class,
                "confidence": confidence,
                "probabilities": class_probabilities
            }

        except Exception as e:

            return {
                "agent": self.name,
                "status": "error",
                "prediction": None,
                "confidence": 0.0,
                "error": str(e)
            } 
