# =============================================================================
# LiverAI Multi-Agent
# Tumor Classification Agent
# =============================================================================

import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import models, transforms


class TumorClassificationAgent:
    """
    Agent for liver tumor classification using EfficientNet-B0.

    Supported input:
        - image path
        - PIL.Image
        - numpy.ndarray

    Output classes:
        0 -> Angiosarcoma
        1 -> Cholangiocarcinoma
        2 -> Healthy
        3 -> Hemangioma
        4 -> Hepatocellular Carcinoma
    """

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        model_path,
        device=None,
        class_names=None
    ):

        self.agent_id = "TumorClassificationAgent"

        self.task_type = "tumor_classification"

        self.modality = "2D_image"

        self.model_name = "EfficientNet-B0"

        # ---------------------------------------------------------------------
        # Device
        # ---------------------------------------------------------------------

        if device is None:

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        else:

            self.device = torch.device(device)

        # ---------------------------------------------------------------------
        # Classes
        # ---------------------------------------------------------------------

        if class_names is None:

            self.classes = [
                "Angiosarcoma",
                "Cholangiocarcinoma",
                "Healthy",
                "Hemangioma",
                "Hepatocellular Carcinoma"
            ]

        else:

            self.classes = list(class_names)

        # ---------------------------------------------------------------------
        # Model path
        # ---------------------------------------------------------------------

        self.model_path = model_path

        if not os.path.exists(self.model_path):

            raise FileNotFoundError(
                f"Tumor model not found:\n{self.model_path}"
            )

        # ---------------------------------------------------------------------
        # Load model
        # ---------------------------------------------------------------------

        self.model = self._build_model()

        self.model.to(self.device)

        self.model.eval()

        print("TumorClassificationAgent")
        print("  Model   :", self.model_name)
        print("  Device  :", self.device)
        print("  Classes :", self.classes)
        print("  Path    :", self.model_path)
        print("  Status  : Model loaded successfully")

    # =========================================================================
    # BUILD MODEL
    # =========================================================================

    def _build_model(self):

        # ---------------------------------------------------------------------
        # EfficientNet-B0 architecture
        # ---------------------------------------------------------------------

        model = models.efficientnet_b0(
            weights=None
        )

        # Replace final classifier
        in_features = model.classifier[1].in_features

        model.classifier[1] = nn.Linear(
            in_features,
            len(self.classes)
        )

        # ---------------------------------------------------------------------
        # Load checkpoint
        # ---------------------------------------------------------------------

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False
        )

        # -------------------------------------------------------------
        # Case 1: raw state_dict
        # -------------------------------------------------------------

        if isinstance(
            checkpoint,
            (
                dict,
                OrderedDict
            )
        ):

            # Common checkpoint keys
            if "state_dict" in checkpoint:

                state_dict = checkpoint["state_dict"]

            elif "model_state_dict" in checkpoint:

                state_dict = checkpoint["model_state_dict"]

            else:

                state_dict = checkpoint

        else:

            raise TypeError(
                "Unsupported tumor checkpoint format: "
                f"{type(checkpoint)}"
            )

        # ---------------------------------------------------------------------
        # Remove possible prefixes
        # ---------------------------------------------------------------------

        cleaned_state_dict = {}

        for key, value in state_dict.items():

            new_key = key

            if new_key.startswith("module."):

                new_key = new_key[len("module."):]

            if new_key.startswith("model."):

                new_key = new_key[len("model."):]

            cleaned_state_dict[new_key] = value

        # ---------------------------------------------------------------------
        # Load weights
        # ---------------------------------------------------------------------

        missing_keys, unexpected_keys = model.load_state_dict(
            cleaned_state_dict,
            strict=False
        )

        print(
            "  Missing keys    :",
            len(missing_keys)
        )

        print(
            "  Unexpected keys :",
            len(unexpected_keys)
        )

        if len(missing_keys) > 0:

            print(
                "  Missing:",
                missing_keys[:10]
            )

        if len(unexpected_keys) > 0:

            print(
                "  Unexpected:",
                unexpected_keys[:10]
            )

        return model

    # =========================================================================
    # IMAGE PREPROCESSING
    # =========================================================================

    def _preprocess(self, image):

        # ---------------------------------------------------------------------
        # Image path
        # ---------------------------------------------------------------------

        if isinstance(image, str):

            if not os.path.exists(image):

                raise FileNotFoundError(
                    f"Tumor image not found:\n{image}"
                )

            image = Image.open(image).convert("RGB")

        # ---------------------------------------------------------------------
        # PIL image
        # ---------------------------------------------------------------------

        elif isinstance(image, Image.Image):

            image = image.convert("RGB")

        # ---------------------------------------------------------------------
        # NumPy array
        # ---------------------------------------------------------------------

        elif isinstance(image, np.ndarray):

            if image.size == 0:

                raise ValueError(
                    "Tumor image numpy array is empty."
                )

            # Grayscale
            if image.ndim == 2:

                image = Image.fromarray(
                    image
                ).convert("RGB")

            # RGB / RGBA
            elif image.ndim == 3:

                if image.shape[2] == 1:

                    image = np.repeat(
                        image,
                        3,
                        axis=2
                    )

                elif image.shape[2] == 4:

                    image = image[:, :, :3]

                elif image.shape[2] != 3:

                    raise ValueError(
                        "Tumor numpy image must have "
                        "1, 3, or 4 channels."
                    )

                # Handle floating point arrays
                if np.issubdtype(
                    image.dtype,
                    np.floating
                ):

                    image = np.nan_to_num(
                        image
                    )

                    # If normalized [0,1]
                    if image.max() <= 1.0:

                        image = image * 255.0

                    image = np.clip(
                        image,
                        0,
                        255
                    ).astype(
                        np.uint8
                    )

                else:

                    image = np.clip(
                        image,
                        0,
                        255
                    ).astype(
                        np.uint8
                    )

                image = Image.fromarray(
                    image
                ).convert("RGB")

            else:

                raise ValueError(
                    "Tumor numpy image must have "
                    "shape (H,W), (H,W,1), "
                    "(H,W,3), or (H,W,4)."
                )

        # ---------------------------------------------------------------------
        # Unsupported input
        # ---------------------------------------------------------------------

        else:

            raise ValueError(
                "Tumor input must be:\n"
                "- image path\n"
                "- PIL.Image\n"
                "- numpy.ndarray"
            )

        # ---------------------------------------------------------------------
        # EfficientNet preprocessing
        # ---------------------------------------------------------------------

        transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224)
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406
                    ],

                    std=[
                        0.229,
                        0.224,
                        0.225
                    ]
                )
            ]
        )

        tensor = transform(image)

        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(
            self.device
        )

        return tensor

    # =========================================================================
    # PREDICTION
    # =========================================================================

    @torch.no_grad()
    def predict(self, image):

        start = time.perf_counter()

        # ---------------------------------------------------------------------
        # Preprocess
        # ---------------------------------------------------------------------

        tensor = self._preprocess(
            image
        )

        # ---------------------------------------------------------------------
        # Inference
        # ---------------------------------------------------------------------

        logits = self.model(
            tensor
        )

        probabilities = F.softmax(
            logits,
            dim=1
        )[0]

        confidence, index = torch.max(
            probabilities,
            dim=0
        )

        index = int(
            index.item()
        )

        confidence = float(
            confidence.item()
        )

        # ---------------------------------------------------------------------
        # Probabilities
        # ---------------------------------------------------------------------

        probs = [
            float(x)
            for x in probabilities.detach().cpu().numpy()
        ]

        # ---------------------------------------------------------------------
        # Prediction
        # ---------------------------------------------------------------------

        prediction = self.classes[index]

        # ---------------------------------------------------------------------
        # Uncertainty
        # ---------------------------------------------------------------------

        uncertainty = (
            1.0 - confidence
        )

        # ---------------------------------------------------------------------
        # Latency
        # ---------------------------------------------------------------------

        latency = (
            time.perf_counter() - start
        ) * 1000.0

        # ---------------------------------------------------------------------
        # Class probabilities
        # ---------------------------------------------------------------------

        class_probabilities = {
            name: probability
            for name, probability
            in zip(
                self.classes,
                probs
            )
        }

        # ---------------------------------------------------------------------
        # Final result
        # ---------------------------------------------------------------------

        return {

            "agent":
                self.agent_id,

            "agent_id":
                self.agent_id,

            "task_type":
                self.task_type,

            "modality":
                self.modality,

            "model":
                self.model_name,

            "prediction":
                prediction,

            "class_index":
                index,

            "probability":
                confidence,

            "confidence":
                confidence,

            "uncertainty":
                uncertainty,

            "probabilities":
                probs,

            "class_probabilities":
                class_probabilities,

            "classes":
                self.classes,

            "status":
                "success",

            "device":
                str(self.device),

            "latency_ms":
                latency,

            "quality":
                1.0,

            "missing_data_ratio":
                0.0,

            "details": {
                "input_shape":
                    list(tensor.shape),

                "num_classes":
                    len(self.classes)
            }
        }

    # =========================================================================
    # ALIAS
    # =========================================================================

    def analyze(self, image):

        return self.predict(
            image
        )

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    def health_check(self):

        return {

            "agent":
                self.agent_id,

            "task_type":
                self.task_type,

            "modality":
                self.modality,

            "model":
                self.model_name,

            "model_path":
                self.model_path,

            "model_exists":
                os.path.exists(
                    self.model_path
                ),

            "device":
                str(self.device),

            "num_classes":
                len(self.classes),

            "classes":
                self.classes,

            "status":
                "healthy"
                if self.model is not None
                else "unhealthy"
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":

    MODEL_PATH = (
        "/content/drive/MyDrive/"
        "models/tumor/"
        "efficientnet_b0_best.pth"
    )

    agent = TumorClassificationAgent(
        model_path=MODEL_PATH
    )

    print("\n" + "=" * 70)
    print("TUMOR AGENT HEALTH CHECK")
    print("=" * 70)

    print(
        agent.health_check()
    )

    # -------------------------------------------------------------------------
    # Synthetic test image
    # -------------------------------------------------------------------------

    test_image = np.random.randint(
        0,
        256,
        (224, 224, 3),
        dtype=np.uint8
    )

    print("\n" + "=" * 70)
    print("TUMOR AGENT TEST")
    print("=" * 70)

    result = agent.predict(
        test_image
    )

    print(
        "Status     :",
        result["status"]
    )

    print(
        "Prediction :",
        result["prediction"]
    )

    print(
        "Class index:",
        result["class_index"]
    )

    print(
        "Confidence :",
        result["confidence"]
    )

    print(
        "Uncertainty:",
        result["uncertainty"]
    )

    print(
        "Latency    :",
        result["latency_ms"],
        "ms"
    )

    print(
        "Probabilities:",
        result["class_probabilities"]
    )
