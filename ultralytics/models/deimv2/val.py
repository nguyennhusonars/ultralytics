# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""DEIMv2 validator."""

from __future__ import annotations

from typing import Any

import torch

from ultralytics.data import YOLODataset
from ultralytics.data.augment import Compose, Format, v8_transforms
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import colorstr, ops

__all__ = ("DEIMv2Validator",)


class DEIMv2Dataset(YOLODataset):
    """DEIMv2 dataset class extending YOLODataset.

    Uses scale-fill letterboxing like RT-DETR for consistent input sizes.
    """

    def __init__(self, *args, data=None, **kwargs):
        """Initialize the DEIMv2Dataset."""
        super().__init__(*args, data=data, **kwargs)

    def load_image(self, i, rect_mode=False):
        """Load one image from dataset index 'i'."""
        return super().load_image(i=i, rect_mode=rect_mode)

    def build_transforms(self, hyp=None):
        """Build transformation pipeline."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp, stretch=True)
        else:
            transforms = Compose([])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms


class DEIMv2Validator(DetectionValidator):
    """DEIMv2 validation class.

    Extends DetectionValidator for DEIMv2-specific post-processing.
    """

    def build_dataset(self, img_path, mode="val", batch=None):
        """Build a DEIMv2 dataset for validation."""
        return DEIMv2Dataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
        )

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Return predictions unchanged as DEIMv2 handles scaling in postprocessing."""
        return predn

    def postprocess(self, preds):
        """Apply confidence thresholding to prediction outputs."""
        if not isinstance(preds, (list, tuple)):
            preds = [preds, None]

        bs, _, nd = preds[0].shape
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)
        bboxes *= self.args.imgsz
        outputs = [torch.zeros((0, 6), device=bboxes.device)] * bs
        for i, bbox in enumerate(bboxes):
            bbox = ops.xywh2xyxy(bbox)
            score, cls = scores[i].max(-1)
            pred = torch.cat([bbox, score[..., None], cls[..., None]], dim=-1)
            pred = pred[score.argsort(descending=True)]
            outputs[i] = pred[score > self.args.conf]

        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5]} for x in outputs]
