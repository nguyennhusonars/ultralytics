# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""DEIMv2 model interface for Ultralytics."""

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import DEIMv2DetectionModel
from ultralytics.utils.torch_utils import TORCH_1_11

from .predict import DEIMv2Predictor
from .train import DEIMv2Trainer
from .val import DEIMv2Validator


class DEIMv2(Model):
    """DEIMv2 detection model.

    DEIMv2 is a transformer-based end-to-end object detector with Fine-grained Distribution Refinement.

    Examples:
        >>> from ultralytics.models.deimv2 import DEIMv2
        >>> model = DEIMv2("deimv2-s.yaml")
        >>> results = model("image.jpg")
    """

    def __init__(self, model: str = "deimv2-s.yaml") -> None:
        """Initialize DEIMv2 model."""
        assert TORCH_1_11, "DEIMv2 requires torch>=1.11"
        super().__init__(model=model, task="detect")

    @property
    def task_map(self) -> dict:
        """Return task map for DEIMv2."""
        return {
            "detect": {
                "predictor": DEIMv2Predictor,
                "validator": DEIMv2Validator,
                "trainer": DEIMv2Trainer,
                "model": DEIMv2DetectionModel,
            }
        }
