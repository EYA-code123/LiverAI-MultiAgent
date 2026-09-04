# ============================================================
# Liver Segmentation Agent
# SegResNet 3D
# ============================================================

import os
import numpy as np
import torch
import torch.nn.functional as F

from monai.networks.nets import SegResNet


class LiverSegmentationAgent:
    """
    Liver segmentation agent based on 3D SegResNet.

    Input:
        - 3D liver CT volume (.npy)
        - numpy array
        - torch tensor

    Preprocessing:
        - float32
        - NaN/Inf cleaning
        - min-max normalization
        - resize to 128 x 128 x 64

    Output:
        - binary liver segmentation mask
        - segmentation statistics
        - model confidence/quality indicators
    """

    def __init__(
        self,
        model_path=None,
        device=None,
        target_size=(128, 128, 64),
        threshold=0.5
    ):

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # ----------------------------------------------------
        # Target size
        # ----------------------------------------------------

        self.target_size = tuple(target_size)

        self.threshold = float(threshold)

        # ----------------------------------------------------
        # Default model path
        # ----------------------------------------------------

        if model_path is None:

            model_path = (
                "/content/drive/MyDrive/"
                "Liver Segmentation Agent/models/"
                "SegResNet3D_Liver_best.pth"
            )

        self.model_path = model_path

        # ----------------------------------------------------
        # Check model
        # ----------------------------------------------------

        if not os.path.exists(self.model_path):

            raise FileNotFoundError(
                f"SegResNet model not found:\n"
                f"{self.model_path}"
            )

        # ----------------------------------------------------
        # Build model
        # ----------------------------------------------------

        self.model = SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            init_filters=16,
            dropout_prob=0.2
        ).to(self.device)

        # ----------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device
        )

        # Support both:
        # 1. state_dict directly
        # 2. checkpoint dictionaries

        if isinstance(checkpoint, dict):

            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]

            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]

            elif "model" in checkpoint:
                state_dict = checkpoint["model"]

            else:
                state_dict = checkpoint

        else:
            state_dict = checkpoint

        # ----------------------------------------------------
        # Remove possible prefixes
        # ----------------------------------------------------

        cleaned_state_dict = {}

        for key, value in state_dict.items():

            new_key = key

            if new_key.startswith("module."):
                new_key = new_key[7:]

            if new_key.startswith("model."):
                new_key = new_key[6:]

            cleaned_state_dict[new_key] = value

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        self.model.load_state_dict(
            cleaned_state_dict,
            strict=True
        )

        self.model.eval()

        # ----------------------------------------------------
        # Information
        # ----------------------------------------------------

        self.loaded = True

        print("=" * 70)
        print("LIVER SEGMENTATION AGENT")
        print("=" * 70)
        print("Model        :", self.model_path)
        print("Device       :", self.device)
        print("Target size  :", self.target_size)
        print("Threshold    :", self.threshold)
        print("Model loaded : SUCCESS")
        print("=" * 70)

    # ========================================================
    # Load NPY
    # ========================================================

    def load_volume(self, input_data):

        # ----------------------------------------------------
        # File path
        # ----------------------------------------------------

        if isinstance(input_data, str):

            if not os.path.exists(input_data):

                raise FileNotFoundError(
                    f"Input volume not found:\n"
                    f"{input_data}"
                )

            volume = np.load(input_data)

        # ----------------------------------------------------
        # NumPy array
        # ----------------------------------------------------

        elif isinstance(input_data, np.ndarray):

            volume = input_data

        # ----------------------------------------------------
        # Torch tensor
        # ----------------------------------------------------

        elif torch.is_tensor(input_data):

            volume = input_data.detach().cpu().numpy()

        else:

            raise TypeError(
                "Input must be:\n"
                "- .npy file path\n"
                "- numpy.ndarray\n"
                "- torch.Tensor"
            )

        return volume

    # ========================================================
    # Preprocessing
    # ========================================================

    def preprocess(self, input_data):

        volume = self.load_volume(input_data)

        # ----------------------------------------------------
        # Convert to float32
        # ----------------------------------------------------

        volume = volume.astype(np.float32)

        # ----------------------------------------------------
        # Remove NaN / Inf
        # ----------------------------------------------------

        volume = np.nan_to_num(
            volume,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        # ----------------------------------------------------
        # Verify 3D
        # ----------------------------------------------------

        if volume.ndim != 3:

            raise ValueError(
                f"Expected a 3D volume, "
                f"received shape {volume.shape}"
            )

        original_shape = volume.shape

        # ----------------------------------------------------
        # Min-max normalization
        # ----------------------------------------------------

        volume_min = volume.min()
        volume_max = volume.max()

        if volume_max > volume_min:

            volume = (
                volume - volume_min
            ) / (
                volume_max - volume_min
            )

        else:

            volume = np.zeros_like(volume)

        # ----------------------------------------------------
        # NumPy -> Torch
        #
        # Original:
        # [H, W, D]
        #
        # -> [1, 1, H, W, D]
        # ----------------------------------------------------

        tensor = torch.from_numpy(volume)

        tensor = tensor.unsqueeze(0)
        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(
            self.device,
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        tensor = F.interpolate(
            tensor,
            size=self.target_size,
            mode="trilinear",
            align_corners=False
        )

        return tensor, original_shape

    # ========================================================
    # Segmentation
    # ========================================================

    @torch.no_grad()
    def predict(self, input_data):

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        image, original_shape = self.preprocess(
            input_data
        )

        # ----------------------------------------------------
        # Model inference
        # ----------------------------------------------------

        logits = self.model(image)

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probabilities = torch.sigmoid(logits)

        # ----------------------------------------------------
        # Binary mask
        # ----------------------------------------------------

        prediction = (
            probabilities >= self.threshold
        ).float()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        mean_probability = (
            probabilities.mean().item()
        )

        max_probability = (
            probabilities.max().item()
        )

        min_probability = (
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

        # ----------------------------------------------------
        # Convert prediction to NumPy
        #
        # [1,1,128,128,64]
        # ->
        # [128,128,64]
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        result = {

            "status": "success",

            "agent": "liver_segmentation",

            "model": "SegResNet3D",

            "model_path": self.model_path,

            "device": str(self.device),

            "input_shape": list(original_shape),

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

            "mean_probability": float(
                mean_probability
            ),

            "min_probability": float(
                min_probability
            ),

            "max_probability": float(
                max_probability
            )
        }

        return result

    # ========================================================
    # Run alias
    # ========================================================

    def run(self, input_data):

        return self.predict(input_data)

    # ========================================================
    # Simple test
    # ========================================================

    def test(self, input_data):

        result = self.predict(input_data)

        print("=" * 70)
        print("SEGMENTATION TEST")
        print("=" * 70)

        print(
            "Status          :",
            result["status"]
        )

        print(
            "Agent           :",
            result["agent"]
        )

        print(
            "Model           :",
            result["model"]
        )

        print(
            "Input shape     :",
            result["input_shape"]
        )

        print(
            "Output shape    :",
            result["output_shape"]
        )

        print(
            "Liver voxels    :",
            result["liver_voxels"]
        )

        print(
            "Liver ratio     :",
            f"{result['liver_ratio']:.4f}"
        )

        print(
            "Mean probability:",
            f"{result['mean_probability']:.4f}"
        )

        print(
            "Max probability :",
            f"{result['max_probability']:.4f}"
        )

        print("=" * 70)

        return result


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":

    MODEL_PATH = (
        "/content/drive/MyDrive/"
        "Liver Segmentation Agent/models/"
        "SegResNet3D_Liver_best.pth"
    )

    TEST_VOLUME = (
        "/content/task03_liver/extracted/image/"
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
