# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""DEIMv2 predictor."""

import torch

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class DEIMv2Predictor(BasePredictor):
    """DEIMv2 predictor extending BasePredictor for inference.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.deimv2 import DEIMv2Predictor
        >>> args = dict(model="deimv2-s.pt", source=ASSETS)
        >>> predictor = DEIMv2Predictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def postprocess(self, preds, img, orig_imgs):
        """Postprocess predictions to produce Results objects."""
        if not isinstance(preds, (list, tuple)):
            preds = [preds, None]

        nd = preds[0].shape[-1]
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        results = []
        for bbox, score, orig_img, img_path in zip(bboxes, scores, orig_imgs, self.batch[0]):
            bbox = ops.xywh2xyxy(bbox)
            max_score, cls = score.max(-1, keepdim=True)
            idx = max_score.squeeze(-1) > self.args.conf
            if self.args.classes is not None:
                idx = (cls == torch.tensor(self.args.classes, device=cls.device)).any(1) & idx
            pred = torch.cat([bbox, max_score, cls], dim=-1)[idx]
            pred = pred[pred[:, 4].argsort(descending=True)][: self.args.max_det]
            oh, ow = orig_img.shape[:2]
            pred[..., [0, 2]] *= ow
            pred[..., [1, 3]] *= oh
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def pre_transform(self, im):
        """Pre-transform input images with LetterBox using scale-fill."""
        letterbox = LetterBox(self.imgsz, auto=False, scale_fill=True)
        return [letterbox(image=x) for x in im]
