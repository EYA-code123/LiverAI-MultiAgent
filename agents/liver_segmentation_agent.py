# =============================================================================
# Liver Segmentation Agent
# =============================================================================

import time
import numpy as np
import tensorflow as tf


class LiverSegmentationAgent:

    def __init__(
        self,
        model_path
    ):

        self.name = (
            "LiverSegmentationAgent"
        )

        self.model_name = (
            "SegResNet / U-Net"
        )

        self.model_path = (
            model_path
        )

        self.model = (
            tf.keras.models.load_model(
                model_path,
                compile=False
            )
        )

    # =========================================================================
    # PREPROCESS
    # =========================================================================

    def preprocess(
        self,
        volume
    ):

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

        if volume.size == 0:

            raise ValueError(
                "Empty liver volume"
            )

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

        return volume

    # =========================================================================
    # PREDICT
    # =========================================================================

    def predict(
        self,
        volume
    ):

        start_time = (
            time.perf_counter()
        )

        try:

            if volume is None:

                raise ValueError(
                    "Liver volume is None"
                )

            volume = self.preprocess(
                volume
            )

            # -----------------------------------------------------------------
            # BATCH DIMENSION
            # -----------------------------------------------------------------

            if volume.ndim == 3:

                volume = np.expand_dims(
                    volume,
                    axis=0
                )

            # -----------------------------------------------------------------
            # MODEL
            # -----------------------------------------------------------------

            prediction = (
                self.model.predict(
                    volume,
                    verbose=0
                )
            )

            # -----------------------------------------------------------------
            # MASK
            # -----------------------------------------------------------------

            binary_mask = (
                prediction > 0.5
            ).astype(
                np.uint8
            )

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

            liver_percentage = (

                liver_voxels
                /
                total_voxels
                *
                100.0

                if total_voxels > 0

                else 0.0
            )

            # -----------------------------------------------------------------
            # SEGMENTATION CONFIDENCE
            # -----------------------------------------------------------------

            prediction_float = (
                np.asarray(
                    prediction
                )
            )

            confidence = float(
                np.mean(
                    np.maximum(
                        prediction_float,
                        1.0
                        -
                        prediction_float
                    )
                )
            )

            uncertainty = (
                1.0 - confidence
            )

            quality = (
                1.0
                if total_voxels > 0
                else 0.0
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
                    "liver_segmentation",

                "model":
                    self.model_name,

                "status":
                    "success",

                "prediction":
                    "liver_segmented",

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

                "segmentation_available":
                    True,

                "liver_voxels":
                    liver_voxels,

                "liver_percentage":
                    float(
                        liver_percentage
                    ),

                "details": {

                    "task_type":
                        "liver_segmentation",

                    "output_type":
                        "3D_binary_mask",

                    "liver_voxels":
                        liver_voxels,

                    "liver_percentage":
                        float(
                            liver_percentage
                        ),

                    "mask":
                        binary_mask
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
                    "liver_segmentation",

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

                "segmentation_available":
                    False,

                "details": {

                    "task_type":
                        "liver_segmentation"
                },

                "error":
                    str(e)
            }
