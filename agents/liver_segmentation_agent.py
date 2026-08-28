import numpy as np
import tensorflow as tf


class LiverSegmentationAgent:

    def __init__(self, model_path):

        self.name = "Liver Segmentation Agent"
        self.model_name = "SegResNet / U-Net"

        self.model_path = model_path

        self.model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

    def preprocess(self, volume):

        if isinstance(volume, str):

            volume = np.load(volume)

        volume = np.asarray(
            volume,
            dtype=np.float32
        )

        if volume.size == 0:
            raise ValueError(
                "Empty liver volume"
            )

        vmin = np.min(volume)
        vmax = np.max(volume)

        if vmax > vmin:

            volume = (
                volume - vmin
            ) / (
                vmax - vmin
            )

        return volume

    def predict(self, volume):

        try:

            if volume is None:
                raise ValueError(
                    "Liver volume is None"
                )

            volume = self.preprocess(
                volume
            )

            # --------------------------------------------------
            # BATCH DIMENSION
            # --------------------------------------------------

            if volume.ndim == 3:

                volume = np.expand_dims(
                    volume,
                    axis=0
                )

            # --------------------------------------------------
            # MODEL
            # --------------------------------------------------

            prediction = self.model.predict(
                volume,
                verbose=0
            )

            # --------------------------------------------------
            # MASK
            # --------------------------------------------------

            binary_mask = (
                prediction > 0.5
            ).astype(np.uint8)

            liver_voxels = int(
                np.sum(binary_mask)
            )

            total_voxels = int(
                np.prod(binary_mask.shape)
            )

            liver_percentage = (
                liver_voxels /
                total_voxels *
                100
                if total_voxels > 0
                else 0
            )

            return {
                "agent": self.name,
                "model": self.model_name,
                "status": "completed",
                "segmentation_available": True,
                "prediction": "liver_segmented",
                "probability": None,
                "liver_voxels": liver_voxels,
                "liver_percentage":
                    float(liver_percentage),
                "mask": binary_mask
            }

        except Exception as e:

            return {
                "agent": self.name,
                "model": self.model_name,
                "status": "error",
                "segmentation_available": False,
                "prediction": None,
                "probability": None,
                "error": str(e)
            }
