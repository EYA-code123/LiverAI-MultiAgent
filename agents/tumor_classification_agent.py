%cd /content/LiverAI-MultiAgent

from pathlib import Path

tumor_code = r'''
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image


class TumorClassificationAgent:

    def __init__(self, model_path, device=None, class_names=None):
        self.model_path = model_path

        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.classes = class_names or [
            "Angiosarcoma",
            "Cholangiocarcinoma",
            "Healthy",
            "Hemangioma",
            "Hepatocellular Carcinoma"
        ]

        print("=" * 70)
        print("TUMOR CLASSIFICATION AGENT")
        print("=" * 70)
        print(f"Model: {self.model_path}")
        print(f"Device: {self.device}")

        self.model = self._load_model()

        print("Model loaded : SUCCESS")
        print("=" * 70)

    def _load_model(self):

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Tumor model not found: {self.model_path}"
            )

        # IMPORTANT:
        # The model is a PyTorch .pth checkpoint.
        # DO NOT use keras.models.load_model().
        model = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=len(self.classes),
            drop_rate=0.4
        )

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False
        )

        # Handle common checkpoint formats
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
            state_dict = checkpoint.state_dict()

        # Remove common prefixes
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

        missing, unexpected = model.load_state_dict(
            cleaned_state_dict,
            strict=False
        )

        if len(missing) > 0:
            print(f"Warning: {len(missing)} missing keys")

        if len(unexpected) > 0:
            print(f"Warning: {len(unexpected)} unexpected keys")

        model.to(self.device)
        model.eval()

        return model

    def _preprocess(self, image):

        if isinstance(image, Image.Image):
            img = image.convert("RGB")

        elif isinstance(image, np.ndarray):

            arr = image

            # Handle grayscale
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)

            # Handle CHW -> HWC
            elif arr.ndim == 3 and arr.shape[0] in [1, 3]:
                if arr.shape[-1] not in [1, 3]:
                    arr = np.transpose(arr, (1, 2, 0))

            # Handle single channel
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)

            # Normalize safely
            arr = arr.astype(np.float32)

            if arr.max() <= 1.0:
                arr = arr * 255.0

            arr = np.clip(arr, 0, 255).astype(np.uint8)

            img = Image.fromarray(arr).convert("RGB")

        else:
            raise TypeError(
                "Tumor input must be a PIL Image or numpy array."
            )

        img = img.resize(
            (224, 224),
            Image.Resampling.BILINEAR
        )

        arr = np.asarray(img).astype(np.float32) / 255.0

        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )

        arr = (arr - mean) / std

        tensor = torch.from_numpy(arr)
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    @torch.no_grad()
    def predict(self, image):

        start_time = time.perf_counter()

        tensor = self._preprocess(image)

        logits = self.model(tensor)

        probabilities = F.softmax(logits, dim=1)[0]

        confidence, class_index = torch.max(
            probabilities,
            dim=0
        )

        class_index = int(class_index.item())
        confidence = float(confidence.item())

        probability_list = [
            float(x)
            for x in probabilities.detach().cpu().numpy()
        ]

        prediction = self.classes[class_index]

        uncertainty = 1.0 - confidence

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        return {
            "agent": "TumorClassificationAgent",
            "model": "EfficientNet-B0",
            "task_type": "tumor_classification",
            "prediction": prediction,
            "class_index": class_index,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "probabilities": probability_list,
            "class_probabilities": {
                cls: prob
                for cls, prob in zip(
                    self.classes,
                    probability_list
                )
            },
            "classes": self.classes,
            "status": "success",
            "device": str(self.device),
            "latency_ms": latency_ms,
            "quality": 1.0,
            "missing_data_ratio": 0.0,
            "modality": "2D_image"
        }

    def run(self, image):
        return self.predict(image)

    def __call__(self, image):
        return self.predict(image)
'''

Path("agents/tumor_classification_agent.py").write_text(
    tumor_code,
    encoding="utf-8"
)

print("✓ tumor_classification_agent.py replaced")
