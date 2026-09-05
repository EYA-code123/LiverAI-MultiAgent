%%writefile /content/LiverAI-MultiAgent/agents/tumor_classification_agent.py

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import timm


class TumorClassificationAgent:

    def __init__(
        self,
        model_path,
        device=None,
        class_names=None,
        image_size=224,
        confidence_threshold=0.5
    ):
        self.model_path = model_path

        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.image_size = image_size
        self.confidence_threshold = confidence_threshold

        self.class_names = class_names or [
            "Angiosarcoma",
            "Cholangiocarcinoma",
            "Healthy",
            "Hemangioma",
            "Hepatocellular Carcinoma"
        ]

        print("=" * 70)
        print("TUMOR CLASSIFICATION AGENT")
        print("=" * 70)
        print(f"Model : {self.model_path}")
        print(f"Device: {self.device}")
        print(f"Classes: {self.class_names}")

        self.model = self._load_model()

        print("✓ Tumor model loaded successfully")
        print("=" * 70)

    # ============================================================
    # MODEL LOADING
    # ============================================================

    def _load_model(self):

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Tumor model not found:\n{self.model_path}"
            )

        # --------------------------------------------------------
        # Create EfficientNet-B0 architecture
        # --------------------------------------------------------

        model = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=len(self.class_names),
            drop_rate=0.4
        )

        # --------------------------------------------------------
        # Load PyTorch checkpoint
        # --------------------------------------------------------

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False
        )

        # --------------------------------------------------------
        # Detect checkpoint structure
        # --------------------------------------------------------

        if isinstance(checkpoint, dict):

            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]

            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]

            elif "model" in checkpoint and isinstance(
                checkpoint["model"], dict
            ):
                state_dict = checkpoint["model"]

            else:
                state_dict = checkpoint

        else:
            state_dict = checkpoint

        # --------------------------------------------------------
        # Clean prefixes
        # --------------------------------------------------------

        cleaned_state_dict = {}

        for key, value in state_dict.items():

            new_key = key

            for prefix in [
                "module.",
                "model.",
                "net.",
                "backbone."
            ]:

                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]

            cleaned_state_dict[new_key] = value

        # --------------------------------------------------------
        # Load weights
        # --------------------------------------------------------

        missing_keys, unexpected_keys = model.load_state_dict(
            cleaned_state_dict,
            strict=False
        )

        print(f"Missing keys    : {len(missing_keys)}")
        print(f"Unexpected keys : {len(unexpected_keys)}")

        if len(missing_keys) > 0:
            print("Missing:")
            for key in missing_keys[:10]:
                print("  ", key)

        if len(unexpected_keys) > 0:
            print("Unexpected:")
            for key in unexpected_keys[:10]:
                print("  ", key)

        model.to(self.device)
        model.eval()

        return model

    # ============================================================
    # IMAGE PREPROCESSING
    # ============================================================

    def _preprocess_image(self, image):

        # --------------------------------------------------------
        # PIL image
        # --------------------------------------------------------

        if isinstance(image, Image.Image):

            image = image.convert("RGB")

            image = image.resize(
                (self.image_size, self.image_size)
            )

            image = np.asarray(
                image,
                dtype=np.float32
            ) / 255.0

        # --------------------------------------------------------
        # NumPy image
        # --------------------------------------------------------

        elif isinstance(image, np.ndarray):

            image = image.astype(np.float32)

            # Remove singleton dimensions
            image = np.squeeze(image)

            # ----------------------------------------------------
            # Grayscale
            # ----------------------------------------------------

            if image.ndim == 2:

                image_min = image.min()
                image_max = image.max()

                if image_max > image_min:
                    image = (
                        image - image_min
                    ) / (
                        image_max - image_min
                    )

                image = np.stack(
                    [image, image, image],
                    axis=-1
                )

            # ----------------------------------------------------
            # H x W x C
            # ----------------------------------------------------

            elif image.ndim == 3:

                # C x H x W -> H x W x C
                if image.shape[0] in [1, 3] and image.shape[-1] not in [1, 3]:
                    image = np.transpose(
                        image,
                        (1, 2, 0)
                    )

                if image.shape[-1] == 1:
                    image = np.repeat(
                        image,
                        3,
                        axis=-1
                    )

                elif image.shape[-1] != 3:
                    raise ValueError(
                        f"Unsupported image shape: {image.shape}"
                    )

            else:

                raise ValueError(
                    f"Unsupported image dimensions: {image.shape}"
                )

            # Normalize
            image_min = image.min()
            image_max = image.max()

            if image_max > image_min:

                if image_max > 1.0:
                    image = image / 255.0

            image = np.clip(
                image,
                0.0,
                1.0
            )

            image = Image.fromarray(
                (image * 255).astype(np.uint8)
            )

            image = image.resize(
                (self.image_size, self.image_size)
            )

            image = np.asarray(
                image,
                dtype=np.float32
            ) / 255.0

        else:

            raise TypeError(
                "Image must be a PIL.Image.Image "
                "or numpy.ndarray."
            )

        # --------------------------------------------------------
        # ImageNet normalization
        # --------------------------------------------------------

        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )

        image = (
            image - mean
        ) / std

        # HWC -> CHW
        image = np.transpose(
            image,
            (2, 0, 1)
        )

        tensor = torch.tensor(
            image,
            dtype=torch.float32
        ).unsqueeze(0)

        return tensor.to(self.device)

    # ============================================================
    # PREDICTION
    # ============================================================

    @torch.no_grad()
    def predict(self, image):

        start_time = time.time()

        tensor = self._preprocess_image(image)

        logits = self.model(tensor)

        probabilities = F.softmax(
            logits,
            dim=1
        )

        confidence, predicted_idx = torch.max(
            probabilities,
            dim=1
        )

        predicted_idx = int(
            predicted_idx.item()
        )

        confidence = float(
            confidence.item()
        )

        probabilities_np = (
            probabilities[0]
            .detach()
            .cpu()
            .numpy()
        )

        prediction = self.class_names[predicted_idx]

        latency_ms = (
            time.time() - start_time
        ) * 1000.0

        uncertainty = 1.0 - confidence

        status = (
            "success"
            if confidence >= self.confidence_threshold
            else "low_confidence"
        )

        return {
            "agent": "TumorClassificationAgent",
            "model": "EfficientNet-B0",
            "task_type": "tumor_classification",

            "prediction": prediction,

            "class_index": predicted_idx,

            "confidence": confidence,

            "uncertainty": uncertainty,

            "probabilities": {
                self.class_names[i]: float(
                    probabilities_np[i]
                )
                for i in range(len(self.class_names))
            },

            "class_probabilities": probabilities_np.tolist(),

            "classes": self.class_names,

            "status": status,

            "device": str(self.device),

            "latency_ms": latency_ms,

            "quality": 1.0,

            "missing_data_ratio": 0.0,

            "modality": "2D_image"
        }

    # ============================================================
    # ORCHESTRATOR COMPATIBILITY
    # ============================================================

    def run(self, image):

        return self.predict(image)

    def __call__(self, image):

        return self.predict(image)

    # ============================================================
    # HEALTH CHECK
    # ============================================================

    def health_check(self):

        return {
            "agent": "TumorClassificationAgent",
            "status": "ready",
            "model_loaded": self.model is not None,
            "model_path": self.model_path,
            "device": str(self.device),
            "classes": self.class_names
        }
