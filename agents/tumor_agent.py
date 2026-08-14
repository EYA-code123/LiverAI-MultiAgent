"""
Liver Tumor Classification Agent
Uses the trained MobileNetV3 model for MRI tumor classification.
"""

import torch
import torch.nn.functional as F


class TumorClassificationAgent:

    def __init__(self, model, device=None, class_names=None):

        self.name = "TumorClassificationAgent"
        self.model_name = "MobileNetV3"

        self.model = model

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)
        self.model.eval()

        self.class_names = class_names

    def predict(self, image_tensor):

        image_tensor = image_tensor.to(self.device)

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        with torch.no_grad():

            outputs = self.model(image_tensor)

            probabilities = F.softmax(outputs, dim=1)

            confidence, predicted_class = torch.max(
                probabilities, dim=1
            )

        class_index = int(predicted_class.item())
        confidence_value = float(confidence.item())

        if self.class_names:
            prediction = self.class_names[class_index]
        else:
            prediction = str(class_index)

        return {
            "agent": self.name,
            "model": self.model_name,
            "prediction": prediction,
            "class_index": class_index,
            "probability": confidence_value,
            "status": "completed"
        }
