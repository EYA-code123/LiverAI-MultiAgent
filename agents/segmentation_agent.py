"""
3D Liver Segmentation Agent
Uses the trained 3D U-Net model for liver segmentation.
"""

import torch


class LiverSegmentationAgent:

    def __init__(self, model, device=None):

        self.name = "LiverSegmentationAgent"
        self.model_name = "3D U-Net"

        self.model = model

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, volume):

        volume = volume.to(self.device)

        if volume.dim() == 4:
            volume = volume.unsqueeze(0)

        with torch.no_grad():

            output = self.model(volume)

            # Binary segmentation
            if output.shape[1] == 1:
                mask = (torch.sigmoid(output) > 0.5).long()

            else:
                mask = torch.argmax(output, dim=1)

        return {
            "agent": self.name,
            "model": self.model_name,
            "segmentation_mask": mask.cpu(),
            "status": "completed"
        }
