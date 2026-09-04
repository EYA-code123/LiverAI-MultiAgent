# =============================================================================
# LiverAI-MultiAgent
# LIVER SEGMENTATION AGENT
# =============================================================================

import os
import time
import traceback

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


class LiverSegmentationAgent:
    """
    3D Liver Segmentation Agent.

    Input:
        3D liver volume as numpy array or .npy file.

    Output:
        Binary 3D liver mask.

    Expected checkpoint:
        PyTorch .pth

    Project dataset:
        Task03 Liver NPY Dataset
    """

    def __init__(
        self,
        model_path,
        device=None,
        threshold=0.5,
    ):

        self.agent_id = "LiverSegmentationAgent"
        self.agent = self.agent_id
        self.task_type = "liver_segmentation"

        self.model_path = model_path

        self.threshold = float(
            threshold
        )

        if torch is None:

            raise ImportError(
                "PyTorch is required for "
                "LiverSegmentationAgent."
            )

        # ---------------------------------------------------------------------
        # DEVICE
        # ---------------------------------------------------------------------

        if device is None:

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        else:

            self.device = torch.device(
                device
            )

        self.model = None

        self._validate_model_path()

        self._load_model()

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def _validate_model_path(self):

        if not os.path.exists(
            self.model_path
        ):

            raise FileNotFoundError(
                "Segmentation model not found:\n"
                f"{self.model_path}"
            )

    # =========================================================================
    # LOAD MODEL
    # =========================================================================

    def _load_model(self):

        print("=" * 70)
        print("LIVER SEGMENTATION AGENT")
        print("=" * 70)

        print(
            "Model:",
            self.model_path
        )

        print(
            "Device:",
            self.device
        )

        try:

            checkpoint = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=False,
            )

            self.model = self._extract_model(
                checkpoint
            )

            if self.model is None:

                raise RuntimeError(
                    "Could not reconstruct the segmentation model "
                    "from the checkpoint."
                )

            self.model.to(
                self.device
            )

            self.model.eval()

            print(
                "✓ Segmentation model loaded"
            )

            print(
                "✓ Device:",
                self.device
            )

        except TypeError:

            # Compatibility with older PyTorch
            checkpoint = torch.load(
                self.model_path,
                map_location=self.device,
            )

            self.model = self._extract_model(
                checkpoint
            )

            if self.model is None:

                raise RuntimeError(
                    "Could not reconstruct the segmentation model."
                )

            self.model.to(
                self.device
            )

            self.model.eval()

            print(
                "✓ Segmentation model loaded"
            )

        except Exception as e:

            print(
                "✗ Failed to load segmentation model."
            )

            print(
                "Error:",
                e
            )

            traceback.print_exc()

            raise

    # =========================================================================
    # EXTRACT MODEL
    # =========================================================================

    def _extract_model(
        self,
        checkpoint
    ):

        # ---------------------------------------------------------------------
        # CASE 1:
        # Entire nn.Module was saved
        # ---------------------------------------------------------------------

        if isinstance(
            checkpoint,
            nn.Module
        ):

            return checkpoint

        # ---------------------------------------------------------------------
        # CASE 2:
        # Dictionary containing full model
        # ---------------------------------------------------------------------

        if isinstance(
            checkpoint,
            dict
        ):

            possible_model_keys = [
                "model",
                "network",
                "net",
                "module",
            ]

            for key in possible_model_keys:

                candidate = checkpoint.get(
                    key
                )

                if isinstance(
                    candidate,
                    nn.Module
                ):

                    return candidate

            # -----------------------------------------------------------------
            # CASE 3:
            # state_dict
            # -----------------------------------------------------------------

            state_dict = None

            for key in [
                "state_dict",
                "model_state_dict",
                "network_state_dict",
            ]:

                candidate = checkpoint.get(
                    key
                )

                if isinstance(
                    candidate,
                    dict
                ):

                    state_dict = candidate

                    break

            # Direct state dict
            if state_dict is None:

                if self._looks_like_state_dict(
                    checkpoint
                ):

                    state_dict = checkpoint

            if state_dict is not None:

                return self._rebuild_model_from_state_dict(
                    state_dict
                )

        return None

    # =========================================================================
    # STATE DICT DETECTION
    # =========================================================================

    @staticmethod
    def _looks_like_state_dict(
        obj
    ):

        if not isinstance(
            obj,
            dict
        ):

            return False

        tensor_values = 0

        for value in obj.values():

            if torch.is_tensor(
                value
            ):

                tensor_values += 1

        return tensor_values > 0

    # =========================================================================
    # MODEL RECONSTRUCTION
    # =========================================================================

    def _rebuild_model_from_state_dict(
        self,
        state_dict
    ):

        """
        Reconstruct the most common simple 3D U-Net checkpoint.

        IMPORTANT:
        If your .pth contains ONLY a state_dict and your training notebook
        uses a different architecture, the exact architecture must be copied
        from that notebook.
        """

        cleaned_state_dict = {}

        for key, value in state_dict.items():

            new_key = key

            if new_key.startswith(
                "module."
            ):

                new_key = new_key[
                    len("module.") :
                ]

            cleaned_state_dict[
                new_key
            ] = value

        # Infer input/output channels
        in_channels = 1
        out_channels = 1

        for key, value in cleaned_state_dict.items():

            if (
                isinstance(value, torch.Tensor)
                and
                value.ndim == 5
            ):

                # First Conv3D
                if (
                    "conv" in key.lower()
                    and
                    value.shape[1] > 0
                ):

                    in_channels = int(
                        value.shape[1]
                    )

                    break

        # Infer output channels from final layer candidates
        for key, value in reversed(
            list(
                cleaned_state_dict.items()
            )
        ):

            if (
                isinstance(value, torch.Tensor)
                and
                value.ndim == 5
                and
                value.shape[0] <= 4
            ):

                out_channels = int(
                    value.shape[0]
                )

                break

        model = Simple3DUNet(
            in_channels=in_channels,
            out_channels=out_channels,
        )

        try:

            model.load_state_dict(
                cleaned_state_dict,
                strict=True,
            )

        except RuntimeError:

            # Try non-strict loading to make the error
            # explicit but allow compatible checkpoints.
            missing, unexpected = model.load_state_dict(
                cleaned_state_dict,
                strict=False,
            )

            if missing:

                raise RuntimeError(
                    "The segmentation checkpoint does not match "
                    "the built-in 3D U-Net architecture.\n"
                    f"Missing keys: {missing[:20]}\n"
                    f"Unexpected keys: {unexpected[:20]}"
                )

        return model

    # =========================================================================
    # LOAD VOLUME
    # =========================================================================

    def _load_volume(
        self,
        volume
    ):

        if volume is None:

            raise ValueError(
                "Segmentation agent received no volume."
            )

        # ---------------------------------------------------------------------
        # FILE
        # ---------------------------------------------------------------------

        if isinstance(
            volume,
            (str, os.PathLike)
        ):

            if not os.path.exists(
                volume
            ):

                raise FileNotFoundError(
                    f"Volume not found: {volume}"
                )

            volume = np.load(
                volume
            )

        # ---------------------------------------------------------------------
        # NUMPY
        # ---------------------------------------------------------------------

        elif isinstance(
            volume,
            np.ndarray
        ):

            volume = volume.copy()

        else:

            raise TypeError(
                "Volume must be a numpy array "
                "or .npy file path."
            )

        volume = np.asarray(
            volume,
            dtype=np.float32
        )

        volume = np.squeeze(
            volume
        )

        if volume.ndim != 3:

            raise ValueError(
                "Expected a 3D volume after squeezing. "
                f"Got shape={volume.shape}"
            )

        if not np.isfinite(
            volume
        ).all():

            volume = np.nan_to_num(
                volume,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        # ---------------------------------------------------------------------
        # NORMALIZATION
        # ---------------------------------------------------------------------

        min_value = float(
            volume.min()
        )

        max_value = float(
            volume.max()
        )

        if max_value > min_value:

            volume = (
                volume - min_value
            ) / (
                max_value - min_value
            )

        else:

            volume = np.zeros_like(
                volume,
                dtype=np.float32
            )

        return volume

    # =========================================================================
    # PREDICTION
    # =========================================================================

    def predict(
        self,
        volume
    ):

        start_time = time.perf_counter()

        try:

            if self.model is None:

                raise RuntimeError(
                    "Segmentation model is not loaded."
                )

            volume = self._load_volume(
                volume
            )

            original_shape = tuple(
                volume.shape
            )

            # -----------------------------------------------------------------
            # Tensor
            # -----------------------------------------------------------------

            tensor = torch.from_numpy(
                volume
            )

            # [D,H,W]
            #
            # -> [1,1,D,H,W]

            tensor = tensor.unsqueeze(
                0
            ).unsqueeze(
                0
            )

            tensor = tensor.to(
                self.device
            )

            # -----------------------------------------------------------------
            # INFERENCE
            # -----------------------------------------------------------------

            with torch.no_grad():

                output = self.model(
                    tensor
                )

            output = self._extract_output(
                output
            )

            # -----------------------------------------------------------------
            # RESTORE / PROCESS OUTPUT
            # -----------------------------------------------------------------

            output = output.detach().float()

            output = torch.squeeze(
                output
            )

            if output.ndim != 3:

                raise ValueError(
                    "Expected 3D segmentation output. "
                    f"Got shape={tuple(output.shape)}"
                )

            # -----------------------------------------------------------------
            # PROBABILITY
            # -----------------------------------------------------------------

            if (
                float(output.min()) < 0.0
                or
                float(output.max()) > 1.0
            ):

                probability = torch.sigmoid(
                    output
                )

            else:

                probability = output

            probability = torch.clamp(
                probability,
                0.0,
                1.0,
            )

            # -----------------------------------------------------------------
            # RESIZE IF NEEDED
            # -----------------------------------------------------------------

            if tuple(
                probability.shape
            ) != original_shape:

                probability = torch.nn.functional.interpolate(
                    probability.unsqueeze(0).unsqueeze(0),
                    size=original_shape,
                    mode="trilinear",
                    align_corners=False,
                ).squeeze()

            mask = (
                probability
                >=
                self.threshold
            ).to(
                torch.uint8
            )

            mask_np = (
                mask
                .cpu()
                .numpy()
                .astype(
                    np.uint8
                )
            )

            probability_np = (
                probability
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            # -----------------------------------------------------------------
            # METRICS
            # -----------------------------------------------------------------

            foreground_voxels = int(
                mask_np.sum()
            )

            total_voxels = int(
                mask_np.size
            )

            foreground_ratio = (
                foreground_voxels
                /
                total_voxels
                if total_voxels > 0
                else 0.0
            )

            confidence = float(
                probability_np[
                    mask_np == 1
                ].mean()
            ) if foreground_voxels > 0 else float(
                1.0 -
                probability_np.mean()
            )

            confidence = max(
                0.0,
                min(
                    1.0,
                    confidence
                )
            )

            uncertainty = (
                1.0 -
                confidence
            )

            elapsed_ms = (
                time.perf_counter()
                -
                start_time
            ) * 1000.0

            return {
                "agent_id":
                    self.agent_id,

                "agent":
                    self.agent_id,

                "task_type":
                    self.task_type,

                "status":
                    "completed",

                "prediction":
                    "liver_segmented",

                "probability":
                    confidence,

                "confidence":
                    confidence,

                "uncertainty":
                    uncertainty,

                "quality":
                    1.0,

                "missing_data_ratio":
                    0.0,

                "latency_ms":
                    elapsed_ms,

                "details": {
                    "modality":
                        "CT/MRI",

                    "input_type":
                        "3D volume",

                    "original_shape":
                        list(
                            original_shape
                        ),

                    "mask_shape":
                        list(
                            mask_np.shape
                        ),

                    "threshold":
                        self.threshold,

                    "foreground_voxels":
                        foreground_voxels,

                    "total_voxels":
                        total_voxels,

                    "foreground_ratio":
                        float(
                            foreground_ratio
                        ),

                    "mask":
                        mask_np,

                    "probability_map":
                        probability_np,
                },

                "mask":
                    mask_np,

                "probability_map":
                    probability_np,

                "explanation":
                    (
                        "3D liver segmentation performed "
                        "using the trained PyTorch model."
                    ),

                "error":
                    None,
            }

        except Exception as e:

            elapsed_ms = (
                time.perf_counter()
                -
                start_time
            ) * 1000.0

            traceback.print_exc()

            return {
                "agent_id":
                    self.agent_id,

                "agent":
                    self.agent_id,

                "task_type":
                    self.task_type,

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
                    elapsed_ms,

                "details":
                    {},

                "explanation":
                    None,

                "error":
                    str(e),

                "traceback":
                    traceback.format_exc(),
            }

    # =========================================================================
    # OUTPUT EXTRACTION
    # =========================================================================

    @staticmethod
    def _extract_output(
        output
    ):

        if isinstance(
            output,
            (list, tuple)
        ):

            output = output[0]

        if isinstance(
            output,
            dict
        ):

            for key in [
                "out",
                "output",
                "logits",
                "segmentation",
                "mask",
            ]:

                if key in output:

                    output = output[key]

                    break

        if not torch.is_tensor(
            output
        ):

            raise TypeError(
                "Segmentation model output "
                "is not a torch.Tensor."
            )

        return output


# =============================================================================
# SIMPLE 3D U-NET
# =============================================================================

class DoubleConv3D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm3d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm3d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),
        )

    def forward(
        self,
        x
    ):

        return self.block(
            x
        )


class Simple3DUNet(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
    ):

        super().__init__()

        self.enc1 = DoubleConv3D(
            in_channels,
            32
        )

        self.pool1 = nn.MaxPool3d(
            2
        )

        self.enc2 = DoubleConv3D(
            32,
            64
        )

        self.pool2 = nn.MaxPool3d(
            2
        )

        self.enc3 = DoubleConv3D(
            64,
            128
        )

        self.pool3 = nn.MaxPool3d(
            2
        )

        self.bottleneck = DoubleConv3D(
            128,
            256
        )

        self.up3 = nn.ConvTranspose3d(
            256,
            128,
            kernel_size=2,
            stride=2,
        )

        self.dec3 = DoubleConv3D(
            256,
            128
        )

        self.up2 = nn.ConvTranspose3d(
            128,
            64,
            kernel_size=2,
            stride=2,
        )

        self.dec2 = DoubleConv3D(
            128,
            64
        )

        self.up1 = nn.ConvTranspose3d(
            64,
            32,
            kernel_size=2,
            stride=2,
        )

        self.dec1 = DoubleConv3D(
            64,
            32
        )

        self.out = nn.Conv3d(
            32,
            out_channels,
            kernel_size=1
        )

    @staticmethod
    def _match_size(
        x,
        reference
    ):

        target = reference.shape[
            2:
        ]

        if tuple(
            x.shape[2:]
        ) == tuple(target):

            return x

        x = torch.nn.functional.interpolate(
            x,
            size=target,
            mode="trilinear",
            align_corners=False,
        )

        return x

    def forward(
        self,
        x
    ):

        e1 = self.enc1(
            x
        )

        e2 = self.enc2(
            self.pool1(e1)
        )

        e3 = self.enc3(
            self.pool2(e2)
        )

        b = self.bottleneck(
            self.pool3(e3)
        )

        d3 = self.up3(
            b
        )

        d3 = self._match_size(
            d3,
            e3
        )

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(
            d3
        )

        d2 = self.up2(
            d3
        )

        d2 = self._match_size(
            d2,
            e2
        )

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(
            d2
        )

        d1 = self.up1(
            d2
        )

        d1 = self._match_size(
            d1,
            e1
        )

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(
            d1
        )

        return self.out(
            d1
        )
