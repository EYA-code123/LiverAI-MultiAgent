import torch
import torch.nn as nn
import timm

from PIL import Image
from torchvision import transforms


class TumorClassificationAgent:

    def __init__(self, model_path):

        self.name = "TumorClassificationAgent"

        self.model_path = model_path

        self.classes = [
            "Angiosarcoma",
            "Cholangiocarcinoma",
            "Healthy",
            "Hemangioma",
            "Hepatocellular_Carcinoma"
        ]

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # =====================================================
        # MODEL
        # =====================================================

        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=len(self.classes),
            drop_rate=0.4
        )

        state_dict = torch.load(
            model_path,
            map_location=self.device
        )

        self.model.load_state_dict(
            state_dict
        )

        self.model.to(self.device)

        self.model.eval()

        # =====================================================
        # TRANSFORM
        # =====================================================

        self.transform = transforms.Compose([

            transforms.Resize(
                (224, 224)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        print(
            "✅ TumorClassificationAgent loaded"
        )

        print(
            "Device :",
            self.device
        )

        print(
            "Classes :",
            self.classes
        )

    # =========================================================
    # PREDICT
    # =========================================================

    def predict(self, image):

        # -----------------------------------------------------
        # Image path
        # -----------------------------------------------------

        if isinstance(image, str):

            image = Image.open(
                image
            ).convert("RGB")

        # -----------------------------------------------------
        # PIL image
        # -----------------------------------------------------

        elif isinstance(image, Image.Image):

            image = image.convert("RGB")

        else:

            raise TypeError(
                "image must be a file path or PIL.Image"
            )

        # -----------------------------------------------------
        # Transform
        # -----------------------------------------------------

        tensor = self.transform(
            image
        ).unsqueeze(0)

        tensor = tensor.to(
            self.device
        )

        # -----------------------------------------------------
        # Inference
        # -----------------------------------------------------

        with torch.no_grad():

            outputs = self.model(
                tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            probability, predicted = torch.max(
                probabilities,
                dim=1
            )

        predicted_index = (
            predicted.item()
        )

        probability_value = (
            probability.item()
        )

        predicted_label = (
            self.classes[
                predicted_index
            ]
        )

        # -----------------------------------------------------
        # Confidence
        # -----------------------------------------------------

        confidence = probability_value

        # -----------------------------------------------------
        # Result
        # -----------------------------------------------------

        return {

            "agent":
                self.name,

            "status":
                "completed",

            "prediction":
                predicted_label,

            "class_index":
                predicted_index,

            "probability":
                probability_value,

            "confidence":
                confidence,

            "classes":
                self.classes
        }
