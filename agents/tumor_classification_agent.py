import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import timm

from PIL import Image


class TumorClassificationAgent:
    """
    Tumor Classification Agent
    Model: EfficientNet-B0

    Supported inputs:
        - PIL.Image
        - numpy.ndarray
        - image file path

    Classes:
        0 -> Angiosarcoma
        1 -> Cholangiocarcinoma
        2 -> Healthy
        3 -> Hemangioma
        4 -> Hepatocellular Carcinoma
    """

    def __init__(
        self,
        model_path,
        device=None,
        class_names=None,
    ):

        self.model_path = model_path

        # --------------------------------------------------
        # DEVICE
        # --------------------------------------------------

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # --------------------------------------------------
        # CLASSES
        # --------------------------------------------------

        self.classes = class_names or [
            "Angiosarcoma",
            "Cholangiocarcinoma",
            "Healthy",
            "Hemangioma",
            "Hepatocellular Carcinoma",
        ]

        # --------------------------------------------------
        # CHECK MODEL
        # --------------------------------------------------

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"Tumor model not found:\n{self.model_path}"
            )

        # --------------------------------------------------
        # LOAD MODEL
        # --------------------------------------------------

        self.model = self._load_model()

        print("=" * 70)
        print("TUMOR CLASSIFICATION AGENT")
        print("=" * 70)
        print("Model        :", "EfficientNet-B0")
        print("Model path   :", self.model_path)
        print("Device       :", self.device)
        print("Classes      :", self.classes)
        print("Model loaded : SUCCESS")
        print("=" * 70)

    # ======================================================
    # LOAD MODEL
    # ======================================================

    def _load_model(self):

        model = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=len(self.classes),
            drop_rate=0.4,
        )

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False,
        )

        # --------------------------------------------------
        # EXTRACT STATE DICT
        # --------------------------------------------------

        if isinstance(checkpoint, dict):

            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]

            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]

            elif (
                "model" in checkpoint
                and isinstance(checkpoint["model"], dict)
            ):
                state_dict = checkpoint["model"]

            else:
                state_dict = checkpoint

        else:

            if hasattr(checkpoint, "state_dict"):
                state_dict = checkpoint.state_dict()
            else:
                raise TypeError(
                    "Unsupported tumor checkpoint format."
                )

        # --------------------------------------------------
        # CLEAN PREFIXES
        # --------------------------------------------------

        cleaned_state_dict = {}

        for key, value in state_dict.items():

            new_key = key

            prefixes = (
                "module.",
                "model.",
                "net.",
                "backbone.",
            )

            changed = True

            while changed:
                changed = False

                for prefix in prefixes:

                    if new_key.startswith(prefix):

                        new_key = new_key[len(prefix):]
                        changed = True
                        break

            cleaned_state_dict[new_key] = value

        # --------------------------------------------------
        # LOAD WEIGHTS
        # --------------------------------------------------

        missing, unexpected = model.load_state_dict(
            cleaned_state_dict,
            strict=False,
        )

        print(
            "Missing keys    :",
            len(missing)
        )

        print(
            "Unexpected keys :",
            len(unexpected)
        )

        if len(missing) > 0:
            print(
                "WARNING: model has missing parameters."
            )

        if len(unexpected) > 0:
            print(
                "WARNING: checkpoint contains unexpected parameters."
            )

        model.to(self.device)
        model.eval()

        return model

    # ======================================================
    # PREPROCESS
    # ======================================================

    def _preprocess(self, image):

        # --------------------------------------------------
        # CASE 1: IMAGE PATH
        # --------------------------------------------------

        if isinstance(image, (str, os.PathLike)):

            image_path = os.fspath(image)

            if not os.path.isfile(image_path):
                raise FileNotFoundError(
                    f"Tumor image not found:\n{image_path}"
                )

            img = Image.open(image_path).convert("RGB")

        # --------------------------------------------------
        # CASE 2: PIL IMAGE
        # --------------------------------------------------

        elif isinstance(image, Image.Image):

            img = image.convert("RGB")

        # --------------------------------------------------
        # CASE 3: NUMPY ARRAY
        # --------------------------------------------------

        elif isinstance(image, np.ndarray):

            arr = image

            # [H, W]
            if arr.ndim == 2:

                arr = np.stack(
                    [arr] * 3,
                    axis=-1
                )

            # [C, H, W] -> [H, W, C]
            elif (
                arr.ndim == 3
                and arr.shape[0] in (1, 3)
                and arr.shape[-1] not in (1, 3)
            ):

                arr = np.transpose(
                    arr,
                    (1, 2, 0)
                )

            # [H, W, 1] -> [H, W, 3]
            if (
                arr.ndim == 3
                and arr.shape[-1] == 1
            ):

                arr = np.repeat(
                    arr,
                    3,
                    axis=-1
                )

            if arr.ndim != 3:
                raise ValueError(
                    f"Expected 2D/3D image array, got shape {arr.shape}"
                )

            arr = arr.astype(np.float32)

            # Normalize 0-1 images to 0-255
            if arr.max() <= 1.0:
                arr = arr * 255.0

            arr = np.clip(
                arr,
                0,
                255
            ).astype(np.uint8)

            img = Image.fromarray(arr).convert("RGB")

        # --------------------------------------------------
        # INVALID INPUT
        # --------------------------------------------------

        else:

            raise TypeError(
                "Tumor input must be:"
                "\n- image path"
                "\n- PIL.Image"
                "\n- numpy.ndarray"
            )

        # --------------------------------------------------
        # RESIZE
        # --------------------------------------------------

        img = img.resize(
            (224, 224),
            Image.Resampling.BILINEAR
        )

        # --------------------------------------------------
        # NUMPY
        # --------------------------------------------------

        arr = np.asarray(img).astype(
            np.float32
        ) / 255.0

        # --------------------------------------------------
        # IMAGENET NORMALIZATION
        # --------------------------------------------------

        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )

        arr = (arr - mean) / std

        # --------------------------------------------------
        # HWC -> CHW
        # --------------------------------------------------

        tensor = torch.from_numpy(arr)

        tensor = tensor.permute(
            2,
            0,
            1
        )

        # --------------------------------------------------
        # BATCH
        # --------------------------------------------------

        tensor = tensor.unsqueeze(0)

        return tensor.to(
            self.device,
            dtype=torch.float32
        )

    # ======================================================
    # PREDICT
    # ======================================================

    @torch.no_grad()
    def predict(self, image):

        start = time.perf_counter()

        tensor = self._preprocess(image)

        logits = self.model(tensor)

        probabilities = F.softmax(
            logits,
            dim=1
        )[0]

        confidence, index = torch.max(
            probabilities,
            dim=0
        )

        index = int(index.item())

        confidence = float(
            confidence.item()
        )

        probs = [
            float(x)
            for x in probabilities.detach().cpu().numpy()
        ]

        prediction = self.classes[index]

        latency = (
            time.perf_counter() - start
        ) * 1000.0

        return {
            "agent": "TumorClassificationAgent",
            "model": "EfficientNet-B0",
            "task_type": "tumor_classification",

            "prediction": prediction,
            "class_index": index,

            "confidence": confidence,
            "uncertainty": 1.0 - confidence,

            "probabilities": probs,

            "class_probabilities": {
                name: probability
                for name, probability in zip(
                    self.classes,
                    probs
                )
            },

            "classes": self.classes,

            "status": "success",

            "device": str(self.device),

            "latency_ms": latency,

            "quality": 1.0,
            "missing_data_ratio": 0.0,

            "modality": "2D_image",
        }

    # ======================================================
    # RUN
    # ======================================================

    def run(self, image):

        return self.predict(image)

    # ======================================================
    # CALL
    # ======================================================

    def __call__(self, image):

        return self.predict(image)
