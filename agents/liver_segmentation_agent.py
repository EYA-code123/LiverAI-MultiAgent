import numpy as np
import torch

from monai.networks.nets import SegResNet


class LiverSegmentationAgent:

    def __init__(self, model_path):

        self.name = "Liver Segmentation Agent"

        self.model_path = model_path

        # ==================================================
        # DEVICE
        # ==================================================

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"[Segmentation] Device: "
            f"{self.device}"
        )

        # ==================================================
        # RECREATE SEGRESNET ARCHITECTURE
        # ==================================================

        self.model = SegResNet(

            spatial_dims=3,

            in_channels=1,

            out_channels=1,

            init_filters=16,

            dropout_prob=0.2

        )

        # ==================================================
        # LOAD STATE DICT
        # ==================================================

        state_dict = torch.load(
            model_path,
            map_location=self.device
        )

        self.model.load_state_dict(
            state_dict
        )

        # ==================================================
        # MOVE TO DEVICE
        # ==================================================

        self.model.to(
            self.device
        )

        # ==================================================
        # EVALUATION MODE
        # ==================================================

        self.model.eval()

        print(
            "[Segmentation] "
            "SegResNet loaded successfully"
        )

    # ======================================================
    # PREPROCESS
    # ======================================================

    def preprocess(self, volume):

        if isinstance(
            volume,
            str
        ):

            volume = np.load(
                volume
            )

        volume = np.asarray(
            volume,
            dtype=np.float32
        )

        # ==================================================
        # NORMALIZATION
        # ==================================================

        vmin = np.min(
            volume
        )

        vmax = np.max(
            volume
        )

        if vmax > vmin:

            volume = (
                volume - vmin
            ) / (
                vmax - vmin
            )

        # ==================================================
        # CONVERT TO TENSOR
        # ==================================================

        tensor = torch.from_numpy(
            volume
        )

        # Expected:
        #
        # [B, C, D, H, W]
        #

        if tensor.ndim == 3:

            tensor = tensor.unsqueeze(
                0
            )

            tensor = tensor.unsqueeze(
                0
            )

        elif tensor.ndim == 4:

            tensor = tensor.unsqueeze(
                0
            )

        else:

            raise ValueError(
                f"Unexpected volume shape: "
                f"{tuple(tensor.shape)}"
            )

        tensor = tensor.to(
            self.device
        )

        return tensor

    # ======================================================
    # PREDICT
    # ======================================================

    def predict(self, volume):

        try:

            # ------------------------------------------------
            # PREPROCESS
            # ------------------------------------------------

            x = self.preprocess(
                volume
            )

            # ------------------------------------------------
            # INFERENCE
            # ------------------------------------------------

            with torch.no_grad():

                logits = self.model(
                    x
                )

                probabilities = torch.sigmoid(
                    logits
                )

            # ------------------------------------------------
            # BINARY MASK
            # ------------------------------------------------

            binary_mask = (
                probabilities > 0.5
            ).to(
                torch.uint8
            )

            binary_mask = (
                binary_mask
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # STATISTICS
            # ------------------------------------------------

            liver_voxels = int(
                np.sum(
                    binary_mask
                )
            )

            total_voxels = int(
                np.prod(
                    binary_mask.shape
                )
            )

            percentage = (

                liver_voxels
                /
                total_voxels
                *
                100

                if total_voxels > 0

                else 0.0
            )

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            return {

                "agent":
                    self.name,

                "status":
                    "success",

                "segmentation_available":
                    True,

                "liver_voxels":
                    liver_voxels,

                "liver_percentage":
                    float(
                        percentage
                    ),

                "mask":
                    binary_mask
            }

        except Exception as e:

            return {

                "agent":
                    self.name,

                "status":
                    "error",

                "segmentation_available":
                    False,

                "error":
                    str(e)
            }
