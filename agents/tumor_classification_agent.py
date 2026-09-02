# =============================================================================
# Tumor Classification Agent
# =============================================================================

import time
import numpy as np
import torch
import timm

from PIL import Image
from torchvision import transforms


class TumorClassificationAgent:

    def __init__(
        self,
        model_path
    ):

        self.name = (
            "TumorClassificationAgent"
        )

        self.model_name = (
            "EfficientNet-B0"
        )

        self.model_path = (
            model_path
        )

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

        # ---------------------------------------------------------------------
        # MODEL
        # ---------------------------------------------------------------------

        self.model = timm.create_model(

            "efficientnet_b0",

            pretrained=False,

            num_classes=
                len(self.classes),

            drop_rate=0.4
        )

        state_dict = torch.load(

            model_path,

            map_location=
                self.device
        )

        self.model.load_state_dict(
            state_dict
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        # ---------------------------------------------------------------------
        # TRANSFORM
        # ---------------------------------------------------------------------

        self.transform = (
            transforms.Compose([

                transforms.Resize(
                    (224, 224)
                ),

                transforms.ToTensor(),

                transforms.Normalize(

                    [0.485, 0.456, 0.406],

                    [0.229, 0.224, 0.225]
                )
            ])
        )

        print(
            "✓ TumorClassificationAgent loaded"
        )

        print(
            "Device:",
            self.device
        )

    # =========================================================================
    # PREDICT
    # =========================================================================

    def predict(
        self,
        image
    ):

        start_time = (
            time.perf_counter()
        )

        try:

            # -----------------------------------------------------------------
            # LOAD IMAGE
            # -----------------------------------------------------------------

            if isinstance(
                image,
                str
            ):

                image = Image.open(
                    image
                ).convert("RGB")

            elif isinstance(
                image,
                Image.Image
            ):

                image = image.convert(
                    "RGB"
                )

            else:

                raise TypeError(
                    "image must be "
                    "a path or PIL.Image"
                )

            # -----------------------------------------------------------------
            # TRANSFORM
            # -----------------------------------------------------------------

            tensor = (
                self.transform(
                    image
                )
                .unsqueeze(0)
                .to(self.device)
            )

            # -----------------------------------------------------------------
            # INFERENCE
            # -----------------------------------------------------------------

            with torch.no_grad():

                outputs = (
                    self.model(
                        tensor
                    )
                )

                probabilities = (
                    torch.softmax(
                        outputs,
                        dim=1
                    )
                )

                probability, predicted = (
                    torch.max(
                        probabilities,
                        dim=1
                    )
                )

            predicted_index = (
                predicted.item()
            )

            confidence = (
                probability.item()
            )

            predicted_label = (
                self.classes[
                    predicted_index
                ]
            )

            probability_list = [

                float(x)

                for x in (
                    probabilities[0]
                    .cpu()
                    .numpy()
                )
            ]

            # -----------------------------------------------------------------
            # QUALITY
            # -----------------------------------------------------------------

            quality = 1.0

            uncertainty = (
                1.0 - confidence
            )

            latency_ms = (
                time.perf_counter()
                -
                start_time
            ) * 1000.0

            return {

                "agent":
                    self.name,

                "task_type":
                    "tumor_classification",

                "model":
                    self.model_name,

                "status":
                    "success",

                "prediction":
                    predicted_label,

                "probability":
                    confidence,

                "confidence":
                    confidence,

                "uncertainty":
                    uncertainty,

                "quality":
                    quality,

                "missing_data_ratio":
                    0.0,

                "latency_ms":
                    latency_ms,

                "class_index":
                    predicted_index,

                "class_probabilities":
                    probability_list,

                "details": {

                    "task_type":
                        "tumor_classification",

                    "disease":
                        "liver_tumor",

                    "classes":
                        self.classes,

                    "device":
                        str(
                            self.device
                        )
                },

                "error":
                    None
            }

        except Exception as e:

            latency_ms = (
                time.perf_counter()
                -
                start_time
            ) * 1000.0

            return {

                "agent":
                    self.name,

                "task_type":
                    "tumor_classification",

                "model":
                    self.model_name,

                "status":
                    "error",

                "prediction":
                    None,

                "probability":
                    None,

                "confidence":
                    0.0,

                "uncertainty":
                    1.0,

                "quality":
                    0.0,

                "missing_data_ratio":
                    1.0,

                "latency_ms":
                    latency_ms,

                "details": {

                    "task_type":
                        "tumor_classification",

                    "disease":
                        "liver_tumor"
                },

                "error":
                    str(e)
            }
