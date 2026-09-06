import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from monai.networks.nets import SegResNet


class LiverSegmentationAgent:
    """
    Liver Segmentation Agent
    Model: 3D SegResNet

    Input:
        - .npy volume
        - numpy.ndarray
        - torch.Tensor

    Output:
        - binary liver mask
        - probability map
        - segmentation statistics
    """

    def __init__(
        self,
        model_path=None,
        device=None,
        target_size=(128, 128, 64),
        threshold=0.5,
    ):

        # --------------------------------------------------
        # DEVICE
        # --------------------------------------------------

        if device is None:

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        else:

            self.device = torch.device(device)

        # --------------------------------------------------
        # PARAMETERS
        # --------------------------------------------------

        self.target_size = tuple(target_size)

        self.threshold = float(threshold)

        # --------------------------------------------------
        # MODEL PATH
        # --------------------------------------------------

        if model_path is None:

            model_path = (
                "/content/drive/MyDrive/"
                "Liver Segmentation Agent/"
                "models/"
                "SegResNet3D_Liver_best.pth"
            )

        self.model_path = model_path

        # --------------------------------------------------
        # CHECK MODEL
        # --------------------------------------------------

        if not os.path.isfile(self.model_path):

            raise FileNotFoundError(
                "SegResNet model not found:\n"
                f"{self.model_path}"
            )

        # --------------------------------------------------
        # BUILD MODEL
        # --------------------------------------------------

        self.model = SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            init_filters=16,
            dropout_prob=0.2,
        ).to(self.device)

        # --------------------------------------------------
        # LOAD CHECKPOINT
        # --------------------------------------------------

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
                    "Unsupported SegResNet checkpoint format."
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
            )

            changed = True

            while changed:

                changed = False

                for prefix in prefixes:

                    if new_key.startswith(prefix):

                        new_key = new_key[
                            len(prefix):
                        ]

                        changed = True
                        break

            cleaned_state_dict[new_key] = value

        # --------------------------------------------------
        # LOAD WEIGHTS
        # --------------------------------------------------

        missing, unexpected = (
            self.model.load_state_dict(
                cleaned_state_dict,
                strict=False,
            )
        )

        print("=" * 70)
        print("LIVER SEGMENTATION AGENT")
        print("=" * 70)

        print(
            "Model path      :",
            self.model_path
        )

        print(
            "Device          :",
            self.device
        )

        print(
            "Target size     :",
            self.target_size
        )

        print(
            "Threshold       :",
            self.threshold
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
                "WARNING: missing model parameters."
            )

        if len(unexpected) > 0:

            print(
                "WARNING: unexpected checkpoint parameters."
            )

        self.model.eval()

        self.loaded = True

        print(
            "Model loaded    : SUCCESS"
        )

        print("=" * 70)

    # ======================================================
    # LOAD VOLUME
    # ======================================================

    def load_volume(self, input_data):

        # --------------------------------------------------
        # PATH
        # --------------------------------------------------

        if isinstance(
            input_data,
            (str, os.PathLike)
        ):

            input_path = os.fspath(input_data)

            if not os.path.isfile(input_path):

                raise FileNotFoundError(
                    "Input volume not found:\n"
                    f"{input_path}"
                )

            volume = np.load(input_path)

        # --------------------------------------------------
        # NUMPY
        # --------------------------------------------------

        elif isinstance(
            input_data,
            np.ndarray
        ):

            volume = input_data

        # --------------------------------------------------
        # TORCH
        # --------------------------------------------------

        elif torch.is_tensor(input_data):

            volume = (
                input_data
                .detach()
                .cpu()
                .numpy()
            )

        else:

            raise TypeError(
                "Input must be:"
                "\n- .npy path"
                "\n- numpy.ndarray"
                "\n- torch.Tensor"
            )

        return volume

    # ======================================================
    # PREPROCESS
    # ======================================================

    def preprocess(self, input_data):

        volume = self.load_volume(
            input_data
        )

        volume = volume.astype(
            np.float32
        )

        volume = np.nan_to_num(
            volume,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        # --------------------------------------------------
        # VERIFY 3D
        # --------------------------------------------------

        if volume.ndim != 3:

            raise ValueError(
                f"Expected a 3D volume, "
                f"received shape {volume.shape}"
            )

        original_shape = volume.shape

        # --------------------------------------------------
        # NORMALIZATION
        # --------------------------------------------------

        volume_min = float(
            volume.min()
        )

        volume_max = float(
            volume.max()
        )

        if volume_max > volume_min:

            volume = (
                volume - volume_min
            ) / (
                volume_max - volume_min
            )

        else:

            volume = np.zeros_like(
                volume
            )

        # --------------------------------------------------
        # NUMPY -> TORCH
        # --------------------------------------------------

        tensor = torch.from_numpy(
            volume
        )

        # [H,W,D]
        # ->
        # [1,1,H,W,D]

        tensor = tensor.unsqueeze(0)
        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(
            self.device,
            dtype=torch.float32,
        )

        # --------------------------------------------------
        # RESIZE
        # --------------------------------------------------

        tensor = F.interpolate(
            tensor,
            size=self.target_size,
            mode="trilinear",
            align_corners=False,
        )

        return tensor, original_shape

    # ======================================================
    # PREDICT
    # ======================================================

    @torch.no_grad()
    def predict(self, input_data):

        start = time.perf_counter()

        # --------------------------------------------------
        # PREPROCESS
        # --------------------------------------------------

        image, original_shape = (
            self.preprocess(
                input_data
            )
        )

        # --------------------------------------------------
        # MODEL
        # --------------------------------------------------

        logits = self.model(image)

        # --------------------------------------------------
        # PROBABILITY
        # --------------------------------------------------

        probabilities = torch.sigmoid(
            logits
        )

        # --------------------------------------------------
        # BINARY MASK
        # --------------------------------------------------

        prediction = (
            probabilities >= self.threshold
        ).float()

        # --------------------------------------------------
        # STATISTICS
        # --------------------------------------------------

        mean_probability = float(
            probabilities.mean().item()
        )

        max_probability = float(
            probabilities.max().item()
        )

        min_probability = float(
            probabilities.min().item()
        )

        liver_voxels = int(
            prediction.sum().item()
        )

        total_voxels = int(
            prediction.numel()
        )

        liver_ratio = (
            liver_voxels / total_voxels
            if total_voxels > 0
            else 0.0
        )

        # --------------------------------------------------
        # NUMPY OUTPUT
        # --------------------------------------------------

        prediction_numpy = (
            prediction
            .squeeze(0)
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        probability_numpy = (
            probabilities
            .squeeze(0)
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        latency = (
            time.perf_counter() - start
        ) * 1000.0

        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        return {

            "status": "success",

            "agent": "liver_segmentation",

            "task_type": "liver_segmentation",

            "model": "SegResNet3D",

            "model_path": self.model_path,

            "device": str(
                self.device
            ),

            "input_shape": list(
                original_shape
            ),

            "output_shape": list(
                prediction_numpy.shape
            ),

            "target_size": list(
                self.target_size
            ),

            "threshold": self.threshold,

            "liver_mask": prediction_numpy,

            "probability_map": probability_numpy,

            "liver_voxels": liver_voxels,

            "total_voxels": total_voxels,

            "liver_ratio": float(
                liver_ratio
            ),

            "mean_probability": mean_probability,

            "min_probability": min_probability,

            "max_probability": max_probability,

            "confidence": float(
                max_probability
            ),

            "uncertainty": float(
                1.0 - max_probability
            ),

            "quality": 1.0,

            "missing_data_ratio": 0.0,

            "modality": "3D_CT",

            "latency_ms": latency,
        }

    # ======================================================
    # RUN
    # ======================================================

    def run(self, input_data):

        return self.predict(
            input_data
        )

    # ======================================================
    # TEST
    # ======================================================

    def test(self, input_data):

        result = self.predict(
            input_data
        )

        print("=" * 70)
        print("SEGMENTATION TEST")
        print("=" * 70)

        print(
            "Status           :",
            result["status"]
        )

        print(
            "Agent            :",
            result["agent"]
        )

        print(
            "Model            :",
            result["model"]
        )

        print(
            "Input shape      :",
            result["input_shape"]
        )

        print(
            "Output shape     :",
            result["output_shape"]
        )

        print(
            "Liver voxels     :",
            result["liver_voxels"]
        )

        print(
            "Liver ratio      :",
            f"{result['liver_ratio']:.4f}"
        )

        print(
            "Mean probability :",
            f"{result['mean_probability']:.4f}"
        )

        print(
            "Max probability  :",
            f"{result['max_probability']:.4f}"
        )

        print(
            "Latency          :",
            f"{result['latency_ms']:.2f} ms"
        )

        print("=" * 70)

        return result


# ==========================================================
# STANDALONE TEST
# ==========================================================

if __name__ == "__main__":

    MODEL_PATH = (
        "/content/drive/MyDrive/"
        "Liver Segmentation Agent/"
        "models/"
        "SegResNet3D_Liver_best.pth"
    )

    TEST_VOLUME = (
        "/content/drive/MyDrive/"
        "archive (2)/image/"
        "liver_0_img.npy"
    )

    agent = LiverSegmentationAgent(
        model_path=MODEL_PATH
    )

    result = agent.test(
        TEST_VOLUME
    )

    print(
        "\nSegmentation completed successfully."
    )
