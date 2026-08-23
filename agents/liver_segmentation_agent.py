import os
import numpy as np
import tensorflow as tf


class LiverSegmentationAgent:

    def __init__(self, model_path):
        self.name = "Liver Segmentation Agent"

        self.model_path = model_path

        self.model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

    def preprocess(self, volume):
        """
        Prepare NPY volume for segmentation model.
        """

        if isinstance(volume, str):

            volume = np.load(volume)

        volume = np.asarray(volume, dtype=np.float32)

        # Normalize
        vmin = np.min(volume)
        vmax = np.max(volume)

        if vmax > vmin:
            volume = (volume - vmin) / (vmax - vmin)

        return volume

    def predict(self, volume):

        try:

            volume = self.preprocess(volume)

            # Add batch dimension if necessary
            if volume.ndim == 3:
                volume = np.expand_dims(volume, axis=0)

            prediction = self.model.predict(
                volume,
                verbose=0
            )

            mask = prediction

            # Binary mask
            binary_mask = (mask > 0.5).astype(np.uint8)

            liver_voxels = int(np.sum(binary_mask))

            total_voxels = int(np.prod(binary_mask.shape))

            percentage = (
                liver_voxels / total_voxels * 100
                if total_voxels > 0
                else 0
            )

            return {
                "agent": self.name,
                "status": "success",
                "segmentation_available": True,
                "liver_voxels": liver_voxels,
                "liver_percentage": float(percentage),
                "mask": binary_mask
            }

        except Exception as e:

            return {
                "agent": self.name,
                "status": "error",
                "segmentation_available": False,
                "error": str(e)
            } 
