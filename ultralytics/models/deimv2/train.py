# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""DEIMv2 trainer."""

from __future__ import annotations

from copy import copy

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DEIMv2DetectionModel
from ultralytics.utils import RANK, colorstr

from .val import DEIMv2Dataset, DEIMv2Validator


class DEIMv2Trainer(DetectionTrainer):
    """Trainer class for the DEIMv2 model.

    Extends DetectionTrainer to handle the DEIMv2 architecture.

    Examples:
        >>> from ultralytics.models.deimv2.train import DEIMv2Trainer
        >>> args = dict(model="deimv2-s.yaml", data="coco8.yaml", imgsz=640, epochs=3)
        >>> trainer = DEIMv2Trainer(overrides=args)
        >>> trainer.train()
    """

    def get_model(self, cfg: dict | None = None, weights: str | None = None, verbose: bool = True):
        """Initialize and return a DEIMv2 model for object detection."""
        model = DEIMv2DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        """Build and return a DEIMv2 dataset."""
        return DEIMv2Dataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            prefix=colorstr(f"{mode}: "),
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )

    def get_validator(self):
        """Return a DEIMv2Validator for model validation."""
        self.loss_names = "giou_loss", "cls_loss", "l1_loss"
        return DEIMv2Validator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
